import os
import torch
import numpy as np

from PIL import Image
from .segbase import SegmentationDataset


class SBUSegmentation(SegmentationDataset):
    BASE_DIR = 'SBUTrain4KRecoveredSmall'
    # BASE_DIR = 'SBU-Test'
    NUM_CLASS = 2

    def __init__(self, root='D:\\PyTorch\\BDRAR-master\\Datasets\\SBU', split='train', mode=None, transform=None, **kwargs):
        super(SBUSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        _sbu_root = os.path.join(root, self.BASE_DIR)
        _mask_dir = os.path.join(_sbu_root, 'ShadowMasks')
        _image_dir = os.path.join(_sbu_root, 'ShadowImages')
        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(root, 'splits')
        if split == 'train':
            _split_f = os.path.join(_splits_dir, 'train.txt')
        elif split == 'val':
            _split_f = os.path.join(_splits_dir, 'val.txt')
        elif split == 'test':
            _split_f = os.path.join(_splits_dir, 'test.txt')
        else:
            raise RuntimeError('Unknown dataset split.')

        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n') + ".jpg")
                if os.path.isfile(_image):
                    self.images.append(_image)
                else:
                    print(_image)
                # assert os.path.isfile(_image)
                # self.images.append(_image)
                if split != 'test':
                    _mask = os.path.join(_mask_dir, line.rstrip('\n') + ".png")
                    if os.path.isfile(_mask):
                        self.masks.append(_mask)
                    else:
                        print(_image)
                    # assert os.path.isfile(_mask)
                    # self.masks.append(_mask)

        if split != 'test':
            assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        # synchronized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        return img, mask

    def __len__(self):
        return len(self.images)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target > 0] = 1
        return torch.from_numpy(target).long()

    @property
    def classes(self):
        """Category names."""
        return ('nonshadow', 'shadow')
