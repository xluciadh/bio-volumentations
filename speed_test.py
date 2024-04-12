# coding: utf-8

# ============================================================================================= #
#  Author:       Samuel Šuľan, Lucia Hradecká, Filip Lux                                        #
#  Copyright:    Lucia Hradecká     : lucia.d.hradecka@gmail.com                                #
#                Filip Lux          : lux.filip@gmail.com                                       #
#                                                                                               #
#  MIT License.                                                                                 #
#                                                                                               #
#  Permission is hereby granted, free of charge, to any person obtaining a copy                 #
#  of this software and associated documentation files (the "Software"), to deal                #
#  in the Software without restriction, including without limitation the rights                 #
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell                    #
#  copies of the Software, and to permit persons to whom the Software is                        #
#  furnished to do so, subject to the following conditions:                                     #
#                                                                                               #
#  The above copyright notice and this permission notice shall be included in all               #
#  copies or substantial portions of the Software.                                              #
#                                                                                               #
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR                   #
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,                     #
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE                  #
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER                       #
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,                #
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE                #
#  SOFTWARE.                                                                                    #
# ============================================================================================= #

from bio_volumentations.core.composition import *
from bio_volumentations.augmentations import *

import time

size_sample = [(1, 64, 65, 66), (1, 128, 128, 128), (1, 256, 256, 256), (1, 512, 512, 64)]

size_sample = [(1, 64, 65, 66), (1, 128, 128, 128), (1, 256, 256, 256)]



# None are placeholders
augmentations_to_check = [
    RandomAffineTransform(angle_limit=[22.5, 22.5, 22.5], p=1),
    RandomAffineTransform(angle_limit=[22.5, 22.5, 22.5],
                          translation_limit=[10, 10, 10],
                          scaling_limit=[.2, .2, .2],
                          spacing=[1, 0.5, 2],
                          p=1),
    RandomScale(scaling_limit=(0.75, 1), p=1),
    RandomScale(scaling_limit=(1, 1.5), p=1),
    Scale(scales=0.75, p=1),
    Scale(scales=1.5, p=1)

    # Flip(axes=[1, 2, 3], p=1),
    # GaussianBlur(sigma=0, p=1),
    # GaussianNoise(var_limit=(0.001, 0.1), mean=0, p=1),
    # HistogramEqualization(bins=256, p=1),
    # Normalize(mean=0, std=1, p=1),
    # NormalizeMeanStd(mean=0.1, std=1, p=1),
    # RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.1),
    # RandomFlip(axes_to_choose=None, p=1),
    # RandomGamma(gamma_limit=(0.8, 1.2), p=1),
    # RandomGaussianBlur(max_sigma=0.8, p=1),
    # RandomRotate90(axes=[1, 2, 3] , p=1),
    # Scale(scale_factor=1.5, p=1),
    # Scale(scale_factor=0.75, p=1),"""
]


def get_augmentations(number: int, shape):
    bigger_shape = (shape[1] * 1.5, shape[2] * 1.5, shape[3] * 1.5)
    smaller_shape = (shape[1] * 0.75, shape[2] * 0.75, shape[3] * 0.75)
    augmentations = [
        CenterCrop(shape=bigger_shape, p=1),
        CenterCrop(shape=smaller_shape, p=1),
        RandomCrop(shape=bigger_shape, p=1),
        RandomCrop(shape=smaller_shape, p=1),
        Resize(bigger_shape, p=1),
        Resize(smaller_shape, p=1),
    ]
    return augmentations[number]


def single_transform(iterations, size, augumentation):
    cummulative = 0
    maximum = 0
    for i in range(iterations):
        test = np.random.uniform(low=0, high=1, size=size)
        aug = Compose(transforms=[augumentation], p=1)
        data = {'image': test}
        second_time = time.time()
        aug_data = aug(**data)
        _ = aug_data['image'].shape
        time_spent = time.time() - second_time
        cummulative += time_spent
        if time_spent > maximum:
            maximum = time_spent
    return maximum, cummulative


def augmentation_getter(augmentations, iteration):
    if augmentations[iteration] is not None:
        return augmentations[iteration]


def transformation_speed_benchmark(iterations):
    f = open("./100iterationCorrections_aff.txt", "w")

    for i, augmentation in enumerate(augmentations_to_check):  # random_scale_transform
        for size in size_sample:
            # augumentation = augumentation_getter(augmentations_to_check, i, size)
            aug_name = augmentation.__class__.__name__
            print(aug_name)
            test_sample = np.random.uniform(low=0, high=1, size=size)
            test = test_sample.copy()

            aug = Compose(transforms=[augmentation], p=1)
            data = {'image': test}
            first_time = time.time()
            aug_data = aug(**data)
            first_result = time.time() - first_time
            maximum, cumulative = single_transform(iterations, size, augmentation)
            result_time = cumulative / iterations
            f.write(f"Transform: {aug_name},  Size: {size}, FirstRun: {first_result:.3f}, " +
                    f"Average: {result_time:.3f}, Iterations: {iterations}, maximum: {maximum:.3f} \n")
            print(f"Transform: {aug_name},  Size: {size}, FirstRun: {first_result:.3f}, " +
                    f"Average: {result_time:.3f}, Iterations: {iterations}, maximum: {maximum:.3f} \n")

    f.close()


if __name__ == '__main__':
    transformation_speed_benchmark(10)
