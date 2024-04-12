# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

from bio_volumentations.core.composition import Compose
from bio_volumentations.augmentations.transforms import RandomAffineTransform

import time
import numpy as np


def tst_volumentations_speed():
    total_volumes_to_check = 100
    sizes_list = [
        (1, 64, 64, 64),
        # (1, 128, 128, 128),
        # (1, 256, 256, 256),
        # (1, 512, 512, 64),
    ]

    for size in sizes_list:

        full_list_to_check = [
            RandomAffineTransform(angle_limit=[22.5, 22.5, 22.5], p=1),
        ]

        for f in full_list_to_check:
            name = f.__class__.__name__
            aug1 = Compose([
                f,
            ], p=1.0)

            data = []
            for i in range(total_volumes_to_check):
                data.append(np.random.uniform(low=0.0, high=255, size=size))
            start_time = time.time()
            for i, cube in enumerate(data):
                
                try:
                    cube1 = aug1(image=cube)['image']
                except Exception as e:
                    print('Augmentation error: {}'.format(str(e)))
                    continue
                
            delta = time.time() - start_time
            print('Size: {} Aug: {} Time: {:.2f} sec Per sample: {:.4f} sec'.format(size, name, delta, delta / len(data)))
            # print(f.__dict__)


if __name__ == '__main__':
    tst_volumentations_speed()
