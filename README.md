# [One-stage Deep Edge Detection Based on Dense-scale Feature Fusion and Pixel-level Imbalance Learning](https://ieeexplore.ieee.org/document/9960785)

## Requirements

pytorch >= 1.0

torchvision

## Training & Testing

### Data preparation

- Download the [BSDS500]() and the [NYUDv2](http://vcl.ucsd.edu/hed/nyu/)
- Place the images to "./data/.'

####
- For NYUDv2 dataset, the following command can bu run for data augmentation
```
python ./data/aug.py
```

### Pretrained Models
- Download the pretrained model [bsds](链接：https://pan.baidu.com/s/17l8KLEu5uXNYPv7pLWHezA?pwd=edzi 
提取码：edzi)

### Training and Testing
- Download the pre-trained model vgg16-bn and EfficientNetv2-S
- model.py represents the use of EfficientNetv2-s model
- model_vgg.py represents the use of VGG16-bn model
```
python main.py
```

### Eval
- The evaluation codes are provided in "./eval"(This is the matlab code)

## Acknowledgment



```
@article{xie2017hed,
author = {Xie, Saining and Tu, Zhuowen},
journal = {International Journal of Computer Vision},
number = {1},
pages = {3--18},
title = {Holistically-Nested Edge Detection},
volume = {125},
year = {2017}
}

@article{liu2019richer,
author = {Liu, Yun and Cheng, Ming-Ming and Hu, Xiaowei and Bian, Jia-Wang and Zhang, Le and Bai, Xiang and Tang, Jinhui},
journal = {IEEE Trans. Pattern Anal. Mach. Intell.},
number = {8},
pages = {1939--1946},
publisher = {IEEE},
title = {Richer Convolutional Features for Edge Detection},
volume = {41},
year = {2019}
}

@inproceedings{he2019bi-directional,
author = {He, Jianzhong and Zhang, Shiliang and Yang, Ming and Shan, Yanhu and Huang, Tiejun},
booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
pages = {3828--3837},
title = {Bi-Directional Cascade Network for Perceptual Edge Detection},
year = {2019}
}

@article{huan2021unmixing,
  title={Unmixing convolutional features for crisp edge detection},
  author={Huan, Linxi and Xue, Nan and Zheng, Xianwei and He, Wei and Gong, Jianya and Xia, Gui-Song},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={44},
  number={10},
  pages={6602--6609},
  year={2021},
  publisher={IEEE}
}

```
