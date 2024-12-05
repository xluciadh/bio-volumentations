# ============================================================================================= #
#  Author:       Lucia Hradecká                                                                 #
#  Copyright:    Lucia Hradecká     : lucia.d.hradecka@gmail.com                                #
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
import numpy as np

# bio-volumentations
import bio_volumentations.core.composition as biovol_compose
import bio_volumentations.augmentations as biovol

# TorchIO: requires pytorch, torchIO (pip) (had networkx 2.8.8)
import torchio
# volumentations: require ???
# TODO
# gunpowder: requires gunpowder (pip) (installed networkx 3.2.1)
import gunpowder

# torchvision: requires pytorch, torchvision
import torchvision.transforms.v2 as torchvision_v2
# albumentations: require albumentations
import albumentations


libs = ['biovol', 'torchio', 'volum', 'gunpowder', 'torchvision', 'album']


# image_shape_list = [(1, 256, 256, 256), (3, 256, 256, 256), (1, 256, 256, 256, 10), (3, 256, 256, 256, 10)]
image_shape_list = [(1, 128, 128, 128)]

# num_repeat = 100
num_repeat = 10

augmentations_biovol = [
    biovol.RandomCrop(),
    biovol.Pad(),
    biovol.RandomFlip(axes_to_choose=None, p=1),
    biovol.RandomAffineTransform(angle_limit=[22.5, 22.5, 22.5], translation_limit=[10, 10, 10],
                          scaling_limit=[.2, .2, .2], spacing=[1, 0.5, 2], p=1),
    biovol.RandomGaussianBlur(max_sigma=0.8, p=1),
    biovol.GaussianNoise(var_limit=(0.001, 0.1), mean=0, p=1),
    biovol.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
    biovol.Normalize(mean=0, std=1, p=1),
]

augmentations_torchio = [
    torchio.Crop(p=1),  # center or random? maybe CroporPad?
    torchio.Pad(p=1),  # maybe CroporPad?
    torchio.CropOrPad(p=1),
    torchio.RandomFlip(axes=[], flip_probability=1.0, p=1),
    torchio.RandomAffine(scales=[], degrees=[], translation=[], p=1),  # TODO isotropic
    torchio.RandomBlur(p=1),
    torchio.RandomNoise(p=1),
    # MISSING: brightness contrast
    torchio.ZNormalization(p=1)
]


def get_transformation_list(lib):
    if lib == 'biovol':
        return augmentations_biovol
    if lib == 'torchio':
        return augmentations_torchio
    if lib == 'volum':
        pass
    if lib == 'gunpowder':
        pass
    if lib == 'torchvision':
        pass
    if lib == 'album':
        pass


def init_compose(lib, transform):
    if lib == 'biovol':
        return biovol_compose.Compose(transforms=[transform], p=1)
    if lib == 'torchio':
        return torchio.Compose([transform], p=1)
    if lib == 'volum':
        pass
    if lib == 'gunpowder':
        pass
    if lib == 'torchvision':
        pass
    if lib == 'album':
        pass


def get_input_data(lib, shape):
    # Shape is given as ([C], D, H, W, [T])

    if lib == 'biovol':
        # ([C], D, H, W, [T])
        return {'image': np.random.uniform(low=0, high=1, size=shape)}
    if lib == 'torchio':
        # (C, W, H, D)
        return np.random.uniform(low=0, high=1, size=shape.transpose((0, 3, 2, 1)))
    if lib == 'volum':
        pass
    if lib == 'gunpowder':
        pass
    if lib == 'torchvision':
        pass
    if lib == 'album':
        pass


def transform_data(lib, data, pipeline):
    if lib == 'biovol':
        augm_data = pipeline(**data)
        return augm_data['image'].shape  # do something to enforce performing the action
    if lib == 'torchio':
        pass
    if lib == 'volum':
        pass
    if lib == 'gunpowder':
        pass
    if lib == 'torchvision':
        pass
    if lib == 'album':
        pass


def single_transform(iterations, size, augmentation, lib):
    cumulative = 0
    maximum = 0
    for i in range(iterations):
        test = np.random.uniform(low=0, high=1, size=size)
        transformation_pipeline = init_compose(lib, augmentation)
        data = {'image': test}
        t_0 = time.time()
        _ = transform_data(lib, data, transformation_pipeline)
        time_spent = time.time() - t_0
        cumulative += time_spent
        if time_spent > maximum:
            maximum = time_spent
    return maximum, cumulative


def transformation_speed_benchmark(iterations):
    f = open(f"./runtime-{num_repeat}_iterations-all.txt", "w")

    for lib in libs:
        print(f'*************** LIBRARY {lib} ***************')

        for i, augmentation in enumerate(get_transformation_list(lib)):
            aug_name = augmentation.__class__.__name__
            print(aug_name)

            for shape in image_shape_list:
                # prepare data and transformation pipeline
                transformation_pipeline = init_compose(lib, augmentation)
                data = get_input_data(lib, shape)

                # run once (prepare the environment)
                first_time = time.time()
                _ = transform_data(lib, data, transformation_pipeline)
                first_result = time.time() - first_time

                # run the measured tries
                maximum, cumulative = single_transform(iterations, shape, augmentation, lib)
                result_time = cumulative / iterations

                # log
                log_message = f"Runtime in seconds. " \
                              f"FirstRun: {first_result:.3f}, Average: {result_time:.3f}, Maximum: {maximum:.3f}. " \
                              f"(Library: {lib}, Transform: {aug_name}, Iterations: {iterations}, ImageSize: {shape})\n"
                f.write(log_message)
                print(log_message)

        print()

    f.close()


if __name__ == '__main__':
    transformation_speed_benchmark(num_repeat)
