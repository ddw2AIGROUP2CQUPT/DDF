#!/usr/bin/env python

import math
import cv2
import os
import os.path as osp
import numpy as np

def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def demo():
    """
    Demos the largest_rotated_rect function
    """

    image = cv2.imread("img_5001.png")
    image_height, image_width = image.shape[0:2]

    cv2.imshow("Original Image", image)

    print("Press [enter] to begin the demo")
    print("Press [q] or Escape to quit")

    key = cv2.waitKey(0)
    if key == ord("q") or key == 27:
        exit()

    for i in np.arange(0, 360, 0.5):
        image_orig = np.copy(image)
        image_rotated = rotate_image(image, i)
        image_rotated_cropped = crop_around_center(
            image_rotated,
            *largest_rotated_rect(
                image_width,
                image_height,
                math.radians(i)
            )
        )

        key = cv2.waitKey(2)
        if(key == ord("q") or key == 27):
            exit()

        cv2.imshow("Original Image", image_orig)
        cv2.imshow("Rotated Image", image_rotated)
        cv2.imshow("Cropped Image", image_rotated_cropped)

    print("Done")


if __name__ == "__main__":
    input_dir = r'./data/NYUD/'
    output_dir = r'./data/NYUD_enhance/'

    # Take hha as an example
    train_lst = osp.join(input_dir, 'hha_train.txt')

    f = open(osp.join(input_dir, train_lst))
    trainval_img = []
    trainval_gt = []
    train_lst = []
    train_gt_lst =[]
    for line in f.readlines():
        train_img_tmp, train_gt_tmp = line.strip().split()
        train_img = osp.join(input_dir, train_img_tmp)
        train_gt = osp.join(input_dir, train_gt_tmp)
        trainval_img.append(train_img)
        trainval_gt.append(train_gt)
        train_lst.append(osp.split(train_img_tmp)[-1])
        train_gt_lst.append(osp.split(train_gt_tmp)[-1])

    # trainval_gt = [osp.join(input_dir, 'Edge', 'train', x) for x in train_lst] \
            # + [osp.join(input_dir, 'Edge', 'val', x) for x in val_lst]
    trainval_lst = train_lst

    for angle in np.arange(0, 360, 360.0/16):
        os.makedirs(osp.join(output_dir, 'train', 'GT', '{:.1f}_0'.format(angle)))
        os.makedirs(osp.join(output_dir, 'train', 'GT', '{:.1f}_1'.format(angle)))
        os.makedirs(osp.join(output_dir, 'train', 'HHA', '{:.1f}_0'.format(angle)))
        os.makedirs(osp.join(output_dir, 'train', 'HHA', '{:.1f}_1'.format(angle)))



    for idx in range(len(train_lst)):

        img = cv2.imread(trainval_img[idx])
        gt = cv2.imread(trainval_gt[idx])
        height, width = img.shape[0:2]

        for angle in np.arange(0, 360, 360.0/16):
            # hha_rotated = rotate_image(hha, angle)
            # hha_rotated = crop_around_center(hha_rotated,
            #     *largest_rotated_rect(width, height, math.radians(angle)))
            # hha_rotated_flip = cv2.flip(hha_rotated, 1)
            # cv2.imwrite(osp.join(output_dir, 'train', 'HHA', '{:.1f}_0'.format(
            #     angle), trainval_lst[idx]), hha_rotated)
            # cv2.imwrite(osp.join(output_dir, 'train', 'HHA', '{:.1f}_1'.format(
            #     angle), trainval_lst[idx]), hha_rotated_flip)

            img_rotated = rotate_image(img, angle)
            img_rotated = crop_around_center(img_rotated,
                *largest_rotated_rect(width, height, math.radians(angle)))
            img_rotated = cv2.resize(img_rotated, (width, height))
            img_rotated_flip = cv2.flip(img_rotated, 1)
            # img_rotated_resize_small = cv2.resize(img_rotated, (0, 0), fx=0.75, fy=0.75)
            # img_rotated_flip_resize_small = cv2.resize(img_rotated_flip, (0, 0), fx=0.75, fy=0.75)
            # img_rotated_resize_large = cv2.resize(img_rotated, (0, 0), fx=1.25, fy=1.25)
            # img_rotated_flip_resize_large = cv2.resize(img_rotated_flip, (0, 0), fx=1.25, fy=1.25)
            cv2.imwrite(osp.join(output_dir, 'train', 'HHA', '{:.1f}_0'.format(
                angle), train_lst[idx]), img_rotated)
            cv2.imwrite(osp.join(output_dir, 'train', 'HHA', '{:.1f}_1'.format(
                angle), trainval_lst[idx]), img_rotated_flip)
            # cv2.imwrite(osp.join(output_dir, 'train', 'Images', '0.75', '{:.1f}_0'.format(
            #     angle), train_lst[idx]), img_rotated_resize_small)
            #
            # cv2.imwrite(osp.join(output_dir, 'train', 'Images', '0.75', '{:.1f}_1'.format(
            #     angle), trainval_lst[idx]), img_rotated_flip_resize_small)
            # cv2.imwrite(osp.join(output_dir, 'train', 'Images', '1.25', '{:.1f}_0'.format(
            #     angle), train_lst[idx]), img_rotated_resize_large)
            # cv2.imwrite(osp.join(output_dir, 'train', 'Images', '1.25', '{:.1f}_1'.format(
            #     angle), trainval_lst[idx]), img_rotated_flip_resize_large)
            #
            gt_rotated = rotate_image(gt, angle)
            gt_rotated = crop_around_center(gt_rotated,
                *largest_rotated_rect(width, height, math.radians(angle)))
            gt_rotated = cv2.resize(gt_rotated, (width, height))
            gt_rotated_flip = cv2.flip(gt_rotated, 1)  # 翻转
            # gt_rotated_resize_small = cv2.resize(gt_rotated, (0, 0), fx=0.75, fy=0.75)
            # gt_rotated_flip_resize_small = cv2.resize(gt_rotated_flip, (0, 0), fx=0.75, fy=0.75)
            # gt_rotated_resize_large = cv2.resize(gt_rotated, (0, 0), fx=1.25, fy=1.25)
            # gt_rotated_flip_resize_large = cv2.resize(gt_rotated_flip, (0, 0), fx=1.25, fy=1.25)
            cv2.imwrite(osp.join(output_dir, 'train', 'GT', '{:.1f}_0'.format(
                angle), train_gt_lst[idx]), gt_rotated)
            cv2.imwrite(osp.join(output_dir, 'train', 'GT', '{:.1f}_1'.format(
                angle), train_gt_lst[idx]), gt_rotated_flip)
            # cv2.imwrite(osp.join(output_dir, 'train', 'GT', '0.75', '{:.1f}_0'.format(
            #     angle), train_gt_lst[idx]), gt_rotated_resize_small)
            # cv2.imwrite(osp.join(output_dir, 'train', 'GT', '0.75', '{:.1f}_1'.format(
            #     angle), train_gt_lst[idx]), gt_rotated_flip_resize_small)
            # cv2.imwrite(osp.join(output_dir, 'train', 'GT', '1.25', '{:.1f}_0'.format(
            #     angle), train_gt_lst[idx]), gt_rotated_resize_large)
            # cv2.imwrite(osp.join(output_dir, 'train', 'GT', '1.25', '{:.1f}_1'.format(
            #     angle), train_gt_lst[idx]), gt_rotated_flip_resize_large)

    # with open(osp.join(output_dir, 'hha-train.lst'), 'w') as fid:
    #     for angle in np.arange(0, 360, 360.0/16):
    #         for idx in range(len(trainval_lst)):
    #             fid.write('train/HHA/{:.1f}_0/'.format(angle) + trainval_lst[idx]
    #                 + ' train/GT/{:.1f}_0/'.format(angle) + trainval_lst[idx] + '\n')
    #     for angle in np.arange(0, 360, 360.0/16):
    #         for idx in range(len(trainval_lst)):
    #             fid.write('train/HHA/{:.1f}_1/'.format(angle) + trainval_lst[idx]
    #                 + ' train/GT/{:.1f}_1/'.format(angle) + trainval_lst[idx] + '\n')

    with open(osp.join(output_dir, 'hha-train.lst'), 'w') as fid:
        for angle in np.arange(0, 360, 360.0/16):
            for idx in range(len(train_lst)):
                fid.write('train/HHA/{:.1f}_0/'.format(angle) + train_lst[idx]
                    + ' train/GT/{:.1f}_0/'.format(angle) + train_gt_lst[idx] + '\n')
                # fid.write('train/Images/0.75/{:.1f}_0/'.format(angle) + train_lst[idx]
                #           + ' train/GT/0.75/{:.1f}_0/'.format(angle) + train_gt_lst[idx] + '\n')
                # fid.write('train/Images/1.25/{:.1f}_0/'.format(angle) + train_lst[idx]
                #           + ' train/GT/1.25/{:.1f}_0/'.format(angle) + train_gt_lst[idx] + '\n')
        for angle in np.arange(0, 360, 360.0/16):
            for idx in range(len(train_lst)):
                fid.write('train/HHA/{:.1f}_1/'.format(angle) + train_lst[idx]
                    + ' train/GT/{:.1f}_1/'.format(angle) + train_gt_lst[idx] + '\n')
                # fid.write('train/Images/0.75/{:.1f}_1/'.format(angle) + train_lst[idx]
                #           + ' train/GT/0.75/{:.1f}_1/'.format(angle) + train_gt_lst[idx] + '\n')
                # fid.write('train/Images/1.25/{:.1f}_1/'.format(angle) + train_lst[idx]
                #           + ' train/GT/1.25/{:.1f}_1/'.format(angle) + train_gt_lst[idx] + '\n')