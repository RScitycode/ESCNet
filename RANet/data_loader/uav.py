"""WHU Shadow  Segmentation Dataset."""
import os
import torch
import numpy as np

from PIL import Image
from data_loader.segbase import SegmentationDataset
from torchvision import transforms
import matplotlib.pyplot as plt


class UAVSegmentation(SegmentationDataset):
    """Shadow  Dataset
    """
    # NUM_CLASS = 3

    def __init__(self, root=r'./datasets', split='train', mode=None, transform=None, **kwargs): #使用相对路径时出错
        super(UAVSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        # print(self.root)
        assert os.path.exists(self.root)
        if split == 'train':
            self.images, self.masks,self.yanmos = _get_sbu_pairs(self.root, self.split)
        else:
            self.images,self.masks = _get_sbu_pairs(self.root, self.split)
        print(self.root)
        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        # img.show()
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        # mask = Image.open(self.masks[index])
        mask = Image.open(self.masks[index]).convert('RGB')

        # mask.show()
        # synchrosized transform
        if self.mode == 'train':
            yanmo = Image.open(self.yanmos[index]).convert('L')

            # img, mask = self._joint_transform(img, mask)
            img, mask,yanmo = self._img_transform(img), self._mask_transform(mask),self._mask_transform(yanmo)

            if self.transform is not None:
                img = self.transform(img)
                toTensor = transforms.ToTensor()
                mask = toTensor(mask)
                yanmo = toTensor(yanmo)
            return img, mask,yanmo, os.path.splitext(os.path.basename(self.images[index]))[0]
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
            # img, mask = img, mask
            if self.transform is not None:
                img = self.transform(img)
                toTensor = transforms.ToTensor()
                mask = toTensor(mask)
            return img, mask, os.path.splitext(os.path.basename(self.images[index]))[0]
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)

            if self.transform is not None:
                img = self.transform(img)
                toTensor = transforms.ToTensor()
                mask = toTensor(mask)
            return img, mask, os.path.splitext(os.path.basename(self.images[index]))[0]



    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

# get image and mask folder paths
def _get_sbu_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder,yanmo_folder):
        img_paths = []
        mask_paths = []
        yanmo_paths = []
        for root, _, files in os.walk(img_folder):
            print(root)
            for filename in files:
                if ' ' not in filename:
                    if filename.endswith('.tif'):
                        imgpath = os.path.join(root, filename)
                        maskpath = os.path.join(mask_folder, filename)
                        yanmopath = os.path.join(yanmo_folder, filename)
                        if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                            img_paths.append(imgpath)
                            mask_paths.append(maskpath)
                            yanmo_paths.append(yanmopath)
                        else:
                            print('cannot find the mask or image pr yanmo:', imgpath, maskpath,yanmopath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        if split == 'train':
            return img_paths, mask_paths,yanmo_paths
        else:
            return img_paths, mask_paths

    if split == 'train':
        img_folder = os.path.join(folder, 'train/unshaded')
        mask_folder = os.path.join(folder, 'train/shaded')
        yanmo_folder = os.path.join(folder, 'train/norm')

        img_paths, mask_paths,yanmo_paths  = get_path_pairs(img_folder, mask_folder,yanmo_folder)

        print(img_paths)
        return img_paths, mask_paths,yanmo_paths
    elif split == 'val':
        img_folder = os.path.join(folder, 'val/sunshaded')
        mask_folder = os.path.join(folder, 'val/shaded')
        yanmo_folder = os.path.join(folder, 'val/norm')

        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder,yanmo_folder)
        return img_paths, mask_paths
    else:
        assert split == 'test'

        img_folder = os.path.join(folder, 'test/unshaded')
        mask_folder = os.path.join(folder, 'test/shaded')
        yanmo_folder = os.path.join(folder, 'test/norm')

        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder,yanmo_folder)
        return img_paths, mask_paths


if __name__ == '__main__':
    dataset = UAVSegmentation(base_size=512, crop_size=512)