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
import src.core.composition as biovol_compose
import src.augmentations as biovol

# TorchIO: requires pytorch, torchIO (pip) (had networkx 2.8.8)
import torchio
# volumentations: with opencv (pip)
import volumentations
# # gunpowder: requires gunpowder (pip) (installed networkx 3.2.1), python>=3.10
# import gunpowder
# import funlib.persistence
# raw = gunpowder.ArrayKey('RAW')  # declare arrays to use in the pipeline

# # torchvision: requires pytorch, torchvision
# import torchvision.transforms.v2 as torchvision_v2
# # albumentations: require albumentations
# import albumentations


# libs = ['biovol', 'torchio', 'volum', 'gunpowder', 'torchvision', 'album']
libs = ['volum', 'biovol', 'torchio']
# libs = ['gunpowder']

num_repeat = 100
# num_repeat = 10

if 'gunpowder' in libs[0]:
    out_file_name = f"./runtime-{num_repeat}_iterations-gunpowder.txt"
else:
    out_file_name = f"./runtime-{num_repeat}_iterations-all-1.txt"


# image_shape_list = [(1, 256, 256, 256), (3, 256, 256, 256), (1, 256, 256, 256, 10), (3, 256, 256, 256, 10)]
image_shape_list = [(1, 256, 256, 256)]
# image_shape_list = [(1, 128, 128, 128)]

crop_shape = (100, 100, 100)
pad_size = 6
pad_shape = (140, 140, 140)
rotation_limit = (22.5,) * 3
translation_limit = (10,) * 3
scaling_limit = (0.8, 1.2) * 3
anisotropic_spacing = (1, 0.5, 2)
blur_sigma_limit = 2
noise_sigma_limits = (0.001, 1)
brightness_limit = 0.2
constrast_limit = 0.2


def get_transformation_list(lib):
    if lib == 'biovol':
        # https://biovolumentations.readthedocs.io/latest/bio_volumentations.augmentations.html#
        return [
            biovol.RandomCrop(shape=crop_shape, border_mode='constant', p=1),  # crop or pad
            biovol.Pad(pad_size=pad_size, border_mode='constant', p=1),  # only pad
            biovol.RandomFlip(axes_to_choose=None, p=1),
            biovol.RandomAffineTransform(angle_limit=rotation_limit, translation_limit=translation_limit,
                                         scaling_limit=scaling_limit, p=1),  # isotropic data
            biovol.RandomAffineTransform(angle_limit=rotation_limit, translation_limit=translation_limit,
                                         scaling_limit=scaling_limit, spacing=anisotropic_spacing, p=1),
            # anisotropic data
            biovol.RandomGaussianBlur(max_sigma=blur_sigma_limit, p=1),
            biovol.GaussianNoise(var_limit=noise_sigma_limits, mean=0, p=1),
            biovol.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=constrast_limit, p=1),
            biovol.Normalize(mean=0, std=1, p=1),
        ]
    if lib == 'torchio':
        # https://torchio.readthedocs.io/
        return [
            torchio.CropOrPad(target_shape=crop_shape, padding_mode=0, p=1),  # TODO center or random? no one knows
            torchio.Pad(padding=pad_size, padding_mode=0, p=1),
            torchio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5, p=1),
            torchio.RandomAffine(scales=scaling_limit, degrees=rotation_limit, translation=translation_limit, p=1),
            # MISSING: anisotropic affine transform (their 'isotropic' has a different meaning)
            torchio.RandomBlur(std=blur_sigma_limit, p=1),
            torchio.RandomNoise(mean=0, std=noise_sigma_limits, p=1),
            # MISSING: brightness contrast
            torchio.ZNormalization(p=1)
        ]
    if lib == 'volum':
        # https://github.com/ZFTurbo/volumentations/blob/master/volumentations/augmentations/transforms.py
        # TODO: no generated docs + some classes are not documented at all (even in the code)
        rotate_limit_vol = (-22.5, 22.5)
        return [
            volumentations.RandomCrop(shape=crop_shape, p=1),  # TODO only crop
            volumentations.PadIfNeeded(shape=pad_shape, border_mode='constant', p=1),  # TODO only pad ??
            volumentations.Flip(axis=None, p=1),  # this is random variant (no deterministic exists)
            # MISSING: affine and translation -> using a sequence of rotation and scale at least
            [volumentations.Rotate(x_limit=rotate_limit_vol, y_limit=rotate_limit_vol, z_limit=rotate_limit_vol, p=1),
             volumentations.RandomScale(scaling_limit[:2], p=1)],
            # MISSING: anisotropic affine (or anisotropic anything)
            # MISSING: implementation of Blur ... volumentations.Blur(blur_limit=(3, 7), p=1),  # TODO: unclear meaning of param blur_limit; is it Gaussian? / alternative: GlassBlur - but docs say it's noise + it isn't gaussian blur only
            volumentations.GaussianNoise(var_limit=noise_sigma_limits, mean=0, p=1),
            volumentations.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=constrast_limit,
                                                    p=1),
            volumentations.Normalize(range_norm=False, p=1),
        ]
    if lib == 'gunpowder':
        # https://funkelab.github.io/gunpowder/
        return [
            [gunpowder.Crop(key=raw, roi=gunpowder.Roi((0,) * 4, (1,) + crop_shape)), gunpowder.RandomLocation()], # only random crop  # TODO this probably operates at smaller image from the begining
            gunpowder.Pad(key=raw, size=gunpowder.Coordinate((0,) + (pad_size,) * 3)),  # only pad  # TODO this pads the image but only retrieves the original-sized image at the end
            gunpowder.SimpleAugment(transpose_probs=[0, 0, 0, 0], p=1),  # only mirror=flip
            # MISSING: affine (only have DeformAugment = elastic transform (rotation, scaling, jitter; needs physical units), and deprecated ElasticAugment)
            # gunpowder.DeformAugment(control_point_spacing=gunpowder.Coordinate((1, 16, 16, 16)),
            #                         jitter_sigma=gunpowder.Coordinate((0,)*4),
            #                         scale_interval=scaling_limit[:2], rotate=True, spatial_dims=3, p=1),  # TODO no way to control rotation + throws some error related to voxel size (WTF?)
            # MISSING: affine anisotropic
            # MISSING: blur
            gunpowder.NoiseAugment(array=raw, mode='gaussian', clip=False, p=1), # TODO image should be of type float and within range [-1, 1] or [0, 1]
            gunpowder.IntensityAugment(array=raw, scale_min=1 - constrast_limit, scale_max=1 + constrast_limit,
                                       shift_min=-brightness_limit, shift_max=brightness_limit, z_section_wise=False,
                                       clip=False, p=1),
            # MISSING: normalize to 0 mean and 1 variance ... there's only normalization to [0,1] interval: gunpowder.Normalize(array=raw)
        ]

    # elastic_augment = gp.ElasticAugment(
    #   control_point_spacing=(16, 16),
    #   jitter_sigma=(4.0, 4.0),
    #   rotation_interval=(0, math.pi/2))
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
        if isinstance(transform, list):
            # the "affine" transform
            return volumentations.Compose(transform, p=1.0)
        else:
            return volumentations.Compose([transform], p=1.0)
    if lib == 'gunpowder':
        if isinstance(transform, list):
            return transform[0] + transform[1]  # hardcoded but the easiest way at the moment
        else:
            return transform
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
        shape_torchio = (shape[0], shape[3], shape[2], shape[1])
        return np.random.uniform(low=0, high=1, size=shape_torchio)
    if lib == 'volum':
        # (H, W, D, [C]) or (D, H, W, [C])
        shape_volum = (shape[2], shape[3], shape[1], shape[0])
        return {'image': np.random.uniform(low=0, high=1, size=shape_volum)}
    if lib == 'gunpowder':
        # ([C], [D], H, W)

        # source = gunpowder.ArraySource(key=raw, array=gunpowder.Array(np.random.uniform(low=0, high=1, size=shape),
        #                                                               spec=gunpowder.ArraySpec()))
        source = gunpowder.ArraySource(key=raw,
                                       array=funlib.persistence.Array(np.random.uniform(low=0, high=1, size=shape)))

        # formulate a request for "raw"
        request = gunpowder.BatchRequest()
        request[raw] = gunpowder.Roi((0,)*len(shape), shape)  # an offset and a size
        # request[raw] = gunpowder.Roi((0,)*4, (None,)*4)  # TODO can I get "the whole image"? Why does this fail?

        return {'source': source, 'request': request}
    if lib == 'torchvision':
        pass
    if lib == 'album':
        pass


def transform_data(lib, data, pipeline):
    if lib == 'biovol':
        augm_data = pipeline(**data)
        return augm_data['image'].shape  # do something to enforce performing the action
    if lib == 'torchio':
        return pipeline(data).shape  # do something to enforce performing the action
    if lib == 'volum':
        augm_data = pipeline(**data)
        return augm_data['image'].shape  # do something to enforce performing the action
    if lib == 'gunpowder':
        whole_pipeline = data['source'] + pipeline
        if 'Crop' in str(pipeline):
            data['request'][raw] = gunpowder.Roi((0,)*(1+len(crop_shape)), (1,) + crop_shape)
        with gunpowder.build(whole_pipeline):
            batch = whole_pipeline.request_batch(data['request'])
        return batch[raw].data.shape  # batch[raw].data is np.ndarray - do something to enforce performing the action
    if lib == 'torchvision':
        pass
    if lib == 'album':
        pass


def single_transform(iterations, shape, augmentation, lib):
    cumulative = 0
    maximum = 0

    for i in range(iterations):
        # prepare data and transformation pipeline
        transformation_pipeline = init_compose(lib, augmentation)
        data = get_input_data(lib, shape)

        # run and measure
        t_0 = time.time()
        _ = transform_data(lib, data, transformation_pipeline)
        time_spent = time.time() - t_0

        # accumulate time
        cumulative += time_spent
        if time_spent > maximum:
            maximum = time_spent

    return maximum, cumulative


def transformation_speed_benchmark(iterations):
    f = open(out_file_name, "w")

    for lib in libs:
        print(f'*************** LIBRARY {lib} ***************')

        for i, augmentation in enumerate(get_transformation_list(lib)):
            aug_name = augmentation.__class__.__name__
            if isinstance(augmentation, list):
                aug_name = ''.join(['Composed'] + ['-' + a.__class__.__name__ for a in augmentation])
            print(aug_name)

            for shape in image_shape_list:

                # the first run (prepare the environment)
                transformation_pipeline = init_compose(lib, augmentation)
                data = get_input_data(lib, shape)
                first_time = time.time()
                _ = transform_data(lib, data, transformation_pipeline)
                first_result = time.time() - first_time

                # the measured runs
                maximum, cumulative = single_transform(iterations, shape, augmentation, lib)
                result_time = cumulative / iterations

                # logging
                log_message = f"Runtime in seconds. " \
                              f"FirstRun: {first_result:.3f}, Average: {result_time:.3f}, Maximum: {maximum:.3f}. " \
                              f"(Library: {lib}, Transform: {aug_name}, Iterations: {iterations}, ImageSize: {shape})\n"
                f.write(log_message)
                print(log_message)

        print()

    f.close()


if __name__ == '__main__':
    transformation_speed_benchmark(num_repeat)
