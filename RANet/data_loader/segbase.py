"""Base segmentation dataset"""
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt

__all__ = ['SegmentationDataset']


class SegmentationDataset(object):
    """Segmentation Base Dataset"""

    def __init__(self, root, split, mode, transform, base_size=520, crop_size=480):
        super(SegmentationDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.BILINEAR)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.BILINEAR)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _joint_transform(self, img, mask):
        # resize
        size = (self.crop_size, self.crop_size)
        assert img.size == mask.size
        img = img.resize(size, Image.BILINEAR)
        mask = mask.resize(size, Image.BILINEAR)
        # mask32 = mask.resize((32, 32), Image.NEAREST)
        # mask64 = mask.resize((64, 64), Image.NEAREST)
        # mask128 = mask.resize((128, 128), Image.NEAREST)
        # flip horizontally
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            # mask32 = mask32.transpose(Image.FLIP_LEFT_RIGHT)
            # mask64 = mask64.transpose(Image.FLIP_LEFT_RIGHT)
            # mask128 = mask128.transpose(Image.FLIP_LEFT_RIGHT)
        # flip vertically
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            # mask32 = mask32.transpose(Image.FLIP_TOP_BOTTOM)
            # mask64 = mask64.transpose(Image.FLIP_TOP_BOTTOM)
            # mask128 = mask128.transpose(Image.FLIP_TOP_BOTTOM)
        # rotate by 90
        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
            # mask32 = mask32.transpose(Image.ROTATE_90)
            # mask64 = mask64.transpose(Image.ROTATE_90)
            # mask128 = mask128.transpose(Image.ROTATE_90)
        # gaussian blur
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        # plt.subplot(121)
        # plt.imshow(img)
        # plt.subplot(122)
        # plt.imshow(mask)
        # plt.show()

        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        # mask32, mask64, mask128 = self._mask_transform(mask32), self._mask_transform(mask64), self._mask_transform(mask128)
        # return img, mask, mask32, mask64, mask128
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        # return np.array(mask).astype('int32')
        return np.array(mask)

    def _img_transform(self, yanmo):
        return np.array(yanmo)

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0
