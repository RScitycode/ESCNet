"""
Prepare your own datasets for CycleGAN

You need to create two directories to host images from domain A /path/to/data/trainA and from domain B /path/to/data/trainB.
Then you can train the model with the dataset flag --dataroot /path/to/data.
Optionally, you can create hold-out test datasets at /path/to/data/testA and /path/to/data/testB to test your model on unseen images.

Prepare your own datasets for pix2pix

Pix2pix's training requires paired data. We provide a python script to generate training data in the form of pairs of images {A,B},
where A and B are two different depictions of the same underlying scene.
For example, these might be pairs {label map, photo} or {bw image, color image}. Then we can learn to translate A to B or B to A:

Create folder /path/to/data with subdirectories A and B. A and B should each have their own subdirectories train, val, test, etc.
In /path/to/data/A/train, put training images in style A. In /path/to/data/B/train, put the corresponding images in style B.
Repeat same for other data splits (val, test, etc).

Corresponding images in a pair {A,B} must be the same size and have the same filename,
e.g., /path/to/data/A/train/1.jpg is considered to correspond to /path/to/data/B/train/1.jpg.

Once the data is formatted this way, call:
python datasets/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data
This will combine each pair of images (A,B) into a single image file, ready for training.
Example:h d
        python datasets/combine_A_and_B.py --fold_A datasets/UAV-WG/A --fold_B datasets/UAV-WG/B --fold_AB datasets/UAV-WG
        python datasets/combine_A_and_B.py --fold_A datasets/UAV-WG-jpg/A --fold_B datasets/UAV-WG-jpg/B --fold_AB datasets/UAV-WG-jpg

See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
"""

import os
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser('create image pairs')
# parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='../dataset/50kshoes_edges')
# parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='../dataset/50kshoes_jpg')
# parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='../dataset/test_AB')
# parser.add_argument('--fold_A', help='input directory for image A', type=str, default='./datasets/UAV-WG-fullshadow-jpg/A')
# parser.add_argument('--fold_B', help='input directory for image B', type=str, default='./datasets/UAV-WG-fullshadow-jpg/B')
# parser.add_argument('--fold_AB', help='output directory', type=str, default='./datasets/UAV-WG-fullshadow-jpg')
parser.add_argument('--fold_A', help='input directory for image A', type=str, default=r'./UAV-SC/train/shaded')
parser.add_argument('--fold_B', help='input directory for image B', type=str, default=r'./UAV-SC/train/unshaded')
parser.add_argument('--fold_AB', help='output directory', type=str, default=r'D./UAV-SC/train/')

parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000000)
parser.add_argument('--use_AB', dest='use_AB', help='if true: (0001_A, 0001_B) to (0001_AB)', action='store_true')
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))

splits = os.listdir(args.fold_A)

for sp in splits:
    # img_fold_A = os.path.join(args.fold_A, sp)
    # img_fold_B = os.path.join(args.fold_B, sp)
    img_fold_A = args.fold_A
    img_fold_B = args.fold_B
    img_list = os.listdir(img_fold_A)
    # img_list = os.listdir(args.fold_A)
    if args.use_AB:
        img_list = [img_path for img_path in img_list if '_A.' in img_path]

    num_imgs = min(args.num_imgs, len(img_list))
    print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
    img_fold_AB = os.path.join(args.fold_AB, 'train')
    if not os.path.isdir(img_fold_AB):
        os.makedirs(img_fold_AB)
    # if not os.path.isdir(args.fold_AB):
    #     os.makedirs(args.fold_AB)
    print('split = %s, number of images = %d' % (sp, num_imgs))
    for n in range(num_imgs):
        name_A = img_list[n]
        path_A = os.path.join(img_fold_A, name_A)
        if args.use_AB:
            name_B = name_A.replace('_A.', '_B.')
        else:
            name_B = name_A
        path_B = os.path.join(img_fold_B, name_B)
        if os.path.isfile(path_A) and os.path.isfile(path_B):
            name_AB = name_A
            if args.use_AB:
                name_AB = name_AB.replace('_A.', '.')  # remove _A
            path_AB = os.path.join(img_fold_AB, name_AB)
            im_A = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
            im_B = cv2.imread(path_B, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
            im_AB = np.concatenate([im_A, im_B], 1)
            cv2.imwrite(path_AB, im_AB)
