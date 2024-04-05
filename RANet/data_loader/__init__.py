"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .mscoco import COCOSegmentation
from .cityscapes import CitySegmentation
from .ade import ADE20KSegmentation
from .pascal_voc import VOCSegmentation
from .ucf import UCFSegmentation
# from .sbu import SBUSegmentation
from .sbu_shadow import SBUSegmentation
from .pascal_aug import VOCAugSegmentation
from .whu import WHUSegmentation
from .uav import UAVSegmentation

datasets = {
    'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
    'ucf': UCFSegmentation,
    # 'sbu': SBUSegmentation,
    'sbu_shadow': SBUSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'coco': COCOSegmentation,
    'citys': CitySegmentation,
    'uav': UAVSegmentation
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
