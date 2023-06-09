# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import csv
import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image
from mmseg.utils import get_root_logger

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ShiftDataset(CustomDataset):
    """ShiftDataset dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """

    CLASSES = ( 'unlabeled', 'building', 'fence', 'other', 'pedestrian', 
    'pole', 'road line', 'road', 'sidewalk', 'vegetation', 
    'vehicle', 'wall', 'traffic sign', 'sky', 'ground', 
    'bridge', 'rail track', 'guard rail', 'traffic light', 'static',
     'dynamic', 'water', 'terrain')

    PALETTE = [[0,0,0], [ 70, 70, 70], [100, 40, 40], [ 55, 90, 80], [220, 20, 60], 
    [153, 153, 153], [157, 234, 50], [128, 64, 128], [244, 35, 232], [107, 142, 35], 
    [ 0, 0, 142], [102, 102, 156], [220, 220, 0], [ 70, 130, 180], [ 81, 0, 81],
     [150, 100, 100], [230, 150, 140], [180, 165, 180], [250, 170, 30], [110, 190, 160],
      [170, 120, 50], [ 45, 60, 150], [145, 170, 100]]

    def __init__(self,
                 
    #              , classes=['building', 'fence', 'pedestrian', 'pole', 'road line', 'road', 
    # 'sidewalk', 'vegetation', 'vehicle', 'wall', 'traffic sign', 'sky',
    #  'traffic light', 'terrain'],
                **kwargs):
        super(ShiftDataset, self).__init__(
            img_suffix='_img_front.jpg', seg_map_suffix='_semseg_front.png', classes=['building', 'fence', 'pedestrian', 'pole', 'road line', 'road', 
     'sidewalk', 'vegetation', 'vehicle', 'wall', 'traffic sign', 'sky',
      'traffic light', 'terrain'], **kwargs)
        
    
    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
        # img_infos = img_infos[0:10]        
        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        img_infos_sorted = sorted(img_infos, key=lambda x: x['filename'])
        return img_infos_sorted 

  
