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

import time

from src.bio_volumentations.core.composition import *
from src.bio_volumentations.augmentations import *


size_sample = [(1, 32, 32, 32, 1), (4, 32, 32, 32, 5), (4, 64, 64, 64, 5), (4, 128, 128, 128, 5)]
# size_sample = [(1, 32, 32, 32)]

num_repeat = 100
# num_repeat = 10

augmentations_to_check = [
    AffineTransform(angles=(22.5, 22.5, 22.5), translation=(20, 20, 20), scale=(0.8, 0.8, 0.8), p=1),
    RandomAffineTransform(angle_limit=(22.5, 22.5, 22.5), translation_limit=(20, 20, 20),
                          scaling_limit=(0.8, 0.8, 0.8), p=1),
    Scale(scales=(0.8, 1, 1.25), p=1),
    RandomScale(scaling_limit=(0.8, 1.25), p=1),
    Flip(p=1),
    RandomFlip(p=1),
    RandomRotate90(p=1),
    GaussianBlur(p=1),
    RandomGaussianBlur(p=1),
    GaussianNoise(p=1),
    PoissonNoise(p=1),
    HistogramEqualization(p=1),
    Normalize(p=1),
    NormalizeMeanStd(mean=0.1, std=1, p=1),
    RandomBrightnessContrast(p=0.1),
    RandomGamma(p=1),
]


def single_transform(iterations, size, augmentation):
    times = np.zeros(iterations)
    for i in range(iterations):
        # get data and transformation
        test = np.random.uniform(low=0, high=1, size=size)
        aug = Compose(transforms=[augmentation], p=1)
        data = {'image': test}

        # run and measure
        second_time = time.time()
        aug_data = aug(**data)
        time_spent = time.time() - second_time
        _ = aug_data['image'].shape
        times[i] = time_spent * 1000  # convert from seconds to milliseconds

    return times


def transformation_speed_benchmark(iterations):
    f = open(f'./runtime-{num_repeat}_iterations-biovol.txt', 'w')

    for i, augmentation in enumerate(augmentations_to_check):

        aug_name = augmentation.__class__.__name__
        print(aug_name)

        for size in size_sample:
            test_sample = np.random.uniform(low=0, high=1, size=size)
            test = test_sample.copy()

            aug = Compose(transforms=[augmentation], p=1)
            data = {'image': test}

            # run for the first time to allocate memory etc.
            first_time = time.time()
            aug_data = aug(**data)
            first_result = (time.time() - first_time) * 1000  # convert to milliseconds

            # run the measured runs
            times_arr = single_transform(iterations, size, augmentation)

            # log
            log_message = f'Runtime in seconds. ' \
                          f'FirstRun: {first_result:.3f}, Average: {times_arr.mean():.3f}, ' \
                          f'Maximum: {times_arr.max():.3f}, Std: {times_arr.std():.3f}. ' \
                          f'(Transform: {aug_name}, Iterations: {iterations}, ImageSize: {size})\n'
            f.write(log_message)
            print(log_message)

    f.close()


if __name__ == '__main__':
    transformation_speed_benchmark(num_repeat)
