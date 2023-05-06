import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
import os
import os.path
import time
import argparse
from torch.utils.data import Dataset, DataLoader
import cv2
import datetime
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname
from model import efficientnetv2_s as mymodel
from dataloader import TestFilelist, ImageFilelist, rand_sampler
from function import f_bce_loss

parser = argparse.ArgumentParser(description='Pytorch Training')
parser.add_argument('--bs', default=10, type=int, metavar='BT', help='batch size')
# =============== optimizer
parser.add_argument('--lr', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=3, type=int,
                    metavar='SS', help='learning rate step size')
parser.add_argument('--gamma', '--gm', default=0.1, type=float,
                    help='learning rate decay parameter: Gamma')
parser.add_argument('--maxepoch', default=25, type=int, metavar='N',
                    help='number of total epochs to run')
# =============== misc
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--gpu', default='0', type=str,
                    help='GPU ID')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--tmp', help='tmp folder', default='result_fun0.01bce')
# ================ dataset
parser.add_argument('--train_dir', type=str, default=r'./data/BSDS/')
parser.add_argument('--test_dir', type=str, default=r'./data/BSDS/')
parser.add_argument('--weights', type=str, default='../pre_efficientnetv2-s.pth', help='initial weights path')

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = "4"

THIS_DIR = abspath(dirname(__file__))
TMP = join(THIS_DIR, args.tmp)
if not isdir(TMP):
  os.makedirs(TMP)
print("lr: ", args.lr)


def main():
    args.cuda = True
    # dataset

    test_dataset = TestFilelist(root=args.test_dir, flist='test.lst', data_name='imlist')

    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=8)
    with open('./data/BSDS/test.lst', 'r') as f:
        test_list = f.readlines()
    test_list = [split(i.rstrip())[1] for i in test_list]
    assert len(test_list) == len(test_loader), "%d vs %d" % (len(test_list), len(test_loader))

    model = mymodel().cuda()
    # 如果存在预训练权重则载入
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=lambda storage, loc: storage.cuda(0))  # 加载预训练模型
            # model.load_state_dict(weights_dict['net'])
            model_dict = model.state_dict()
            weights_dict = {k: v for k, v in weights_dict.items()
                                 if k in model_dict}
            model_dict.update(weights_dict)
            model.load_state_dict(model_dict)
            # print(model.load_state_dict(load_weights_dict, strict=False))

        else:
            raise FileNotFoundError("not found weights file : {}".format(args.weights_voc))


    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

    # tensorboard
    from torch.utils.tensorboard import SummaryWriter  # for torch 1.4 or greather
    tb_writer = SummaryWriter(log_dir=args.tmp)

    for epoch in range(args.start_epoch, args.maxepoch):
        print('Epoch {}/{}'.format(epoch, args.maxepoch))
        print('-' * 10)
        rand_sampler()
        train_dataset = ImageFilelist(root=args.train_dir, flist='bsds1.txt',
                                      data_name='imlist')
        train_loader = DataLoader(train_dataset, batch_size=args.bs, num_workers=8, shuffle=True, pin_memory=True)
        train(train_loader, model, optimizer, epoch, tb_writer, save_dir=TMP)
        test(model, test_loader, epoch, test_list=test_list, save_dir=join(TMP, "epoch-%d-testing"% epoch))
        scheduler.step()  # adjust learning rate
        print("lr:", scheduler.get_last_lr())



def train(train_loader, model, optimizer, epoch, tb_writer, save_dir):
    best_loss = 10.0
    ts = time.strftime("%Y%m%d%H%M%S", time.localtime())
    if not isdir(save_dir):
        os.makedirs(save_dir)

    model.train()
    i = 0

    for data in train_loader:
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        optimizer.zero_grad()
        i += 1

        output = model(inputs)


        loss = f_bce_loss().cuda()
        final_loss = loss(output, labels)

        final_loss.backward()
        optimizer.step()

        tb_writer.add_scalar('loss',
                             final_loss.detach(),
                             (len(train_loader) * epoch + i))

        dt = datetime.datetime.now()
        print('[' + dt.strftime('%y-%m-%d %I:%M:%S %p') + '] epoch {}, iteration {}, {} : {:.4f}'.format(epoch, i,
                                                                                                         'final_loss',
                                                                                                         final_loss.data.item()))

    print()

    checkpoint = {
        "epoch": epoch + 1,
        "net": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, join(save_dir, "epoch-%d-checkpoint.pth.tar" % epoch))


def test(model, testloader, epoch, test_list, save_dir):
    model.eval()
    if not isdir(save_dir):
        os.makedirs(save_dir)
    for i, image in enumerate(testloader):
        image = image.cuda()
        _, _, H, W = image.shape
        results = model(image)
        result = torch.squeeze(results[-1].detach()).cpu().numpy()
        # results_all = torch.zeros((len(results), 1, H, W))
        # for o in range(len(results)):
        #     results_all[o, 0, :, :] = results[o]
        filename = splitext(test_list[i])[0]
        # torchvision.utils.save_image(1 - results_all, join(save_dir, "%s.jpg" % filename))
        result = cv2.resize(result, (W + 1, H + 1))
        cv2.imwrite(join(save_dir, "%s.png" % filename), result * 255)
        print("running test [%d/%d]" % (i + 1, len(testloader)))


if __name__ == '__main__':
    main()