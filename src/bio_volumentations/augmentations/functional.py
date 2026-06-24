# ============================================================================================= #
#  Author:       Pavel Iakubovskii, ZFTurbo, ashawkey, Dominik Müller,                          #
#                Samuel Šuľan, Lucia Hradecká, Filip Lux                                        #
#  Copyright:    albumentations:    : https://github.com/albumentations-team                    #
#                Pavel Iakubovskii  : https://github.com/qubvel                                 #
#                ZFTurbo            : https://github.com/ZFTurbo                                #
#                ashawkey           : https://github.com/ashawkey                               #
#                Dominik Müller     : https://github.com/muellerdo                              #
#                Lucia Hradecká     : lucia.d.hradecka@gmail.com                                #
#                Filip Lux          : lux.filip@gmail.com                                       #
#                Samuel Šuľan                                                                   #
#                                                                                               #
#  Volumentations History:                                                                      #
#       - Original:                 https://github.com/albumentations-team/albumentations       #
#       - 3D Conversion:            https://github.com/ashawkey/volumentations                  #
#       - Continued Development:    https://github.com/ZFTurbo/volumentations                   #
#       - Enhancements:             https://github.com/qubvel/volumentations                    #
#       - Further Enhancements:     https://github.com/muellerdo/volumentations                 #
#       - Biomedical Enhancements:  https://gitlab.fi.muni.cz/cbia/bio-volumentations           #
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

import numpy as np
import skimage.transform as skt
from skimage.exposure import equalize_hist
from scipy.ndimage import gaussian_filter
from warnings import warn

from .sitk_utils import get_affine_transform, apply_sitk_transform
from .utils import is_included, atleast_kd
from ..biovol_typing import TypeTripletFloat, TypeSpatioTemporalCoordinate, TypeSextetInt, TypeSpatialShape
from ..random_utils import normal, poisson

"""
vol: [C, D, H, W (, T)]

you should give (D, H, W) form shape.

skimage interpolation notations:

order = 0: Nearest-Neighbor
order = 1: Bi-Linear (default)
order = 2: Bi-Quadratic
order = 3: Bi-Cubic
order = 4: Bi-Quartic
order = 5: Bi-Quintic

Interpolation behaves strangely when input of type int.
** Be sure to change volume and mask data type to float !!! **  (already done by Float() in compose - TODO not for int-mask)

But for parameters use primarily ints.
"""


def resize(img, input_new_shape: TypeSpatialShape, interpolation=1, border_mode='reflect', cval=0, mask=False,
           anti_aliasing_downsample=True):

    if input_new_shape is None:
        return img

    new_shape = list(input_new_shape)

    # shape adjustment (time-lapse data)
    if len(new_shape) < len(img.shape) - 1:
        new_shape = new_shape + [img.shape[-1]]

    # resize mask and fmask
    anti_aliasing = False
    if mask:
        new_img = skt.resize(
            img,
            new_shape,
            order=interpolation,
            mode=border_mode,
            cval=cval,
            clip=True,
            anti_aliasing=anti_aliasing
        )
        return new_img

    # resize image
    if anti_aliasing_downsample and np.any(np.array(img.shape[1:]) < np.array(new_shape)):
        anti_aliasing = True

    data = []
    for i in range(img.shape[0]):  # for each channel
        subimg = img[i].copy()
        d0 = skt.resize(
            subimg,
            new_shape,
            order=interpolation,
            mode=border_mode,
            cval=cval,
            clip=True,
            anti_aliasing=anti_aliasing
        )
        data.append(d0.copy())
    new_img = np.stack(data, axis=0)

    return new_img


def resize_keypoints(keypoints,
                     domain_limit: TypeSpatialShape,
                     new_shape: TypeSpatialShape):
    if new_shape is None:
        return keypoints

    assert len(domain_limit) == len(new_shape) == 3

    # for each dim compute ratio
    ratio = np.array(new_shape) / np.array(domain_limit)

    keys = np.asarray(keypoints)
    if keys.shape[1] == 4:
        ratio = np.append(ratio, 1)  # add the "resizing" constant for the temporal dimension

    return list(map(tuple, keys * ratio))


def affine(img: np.array,
           transform,
           interpolation: str = 'linear',
           value: float = 0,
           spacing: TypeTripletFloat = (1, 1, 1)):
    """Compute affine transformation of a multi-channel image.

    Args:
        img: image data (np.ndarray) in the following format: (channel, Z, Y, X, [time])
        transform: sitk transformation to be applied to the image
        interpolation: interpolation type
        value: default pixel value
        spacing: relative voxel size (in XYZ format)

    Returns:
        np.ndarray: Transformed image (np.ndarray).
    """
    # apply the transformation (we need to change order of axes of spacing to XYZ)
    return apply_sitk_transform(img,
                                sitk_transform=transform,
                                interpolation=interpolation,
                                default_value=value,
                                spacing=spacing)


def affine_keypoints(keypoints: list,
                     transform,
                     domain_limit: TypeSpatioTemporalCoordinate,
                     keep_all: bool = False,
                     ):
    """Compute affine transformation of a set of keypoints.

    Args:
        keypoints: list of input keypoints (in ZYX format)
        transform: sitk transformation to be applied to the keypoints
        domain_limit: limit of the domain (in ZYXT format), there keypoints can appear, it is used to define center of transforms
                and to filter out output key-point from the outside of the domain
        keep_all: True to keep also keypoints outside the image domain

    Returns:
        list: A list of transformed keypoints (in ZYX format).

    """

    transform = transform.GetInverse()

    res = []
    for point in keypoints:
        # we need to change the order of axes of points to XYZ for the transformation and then back to ZYX
        point_spatial = point[:3]
        transformed_point = transform.TransformPoint(point_spatial[::-1])[::-1] + point[3:]
        if keep_all or is_included(domain_limit, transformed_point[:3]):
            res.append(transformed_point)
    return res


# Used in rot90_keypoints
def flip_keypoints(keypoints, axes, img_shape):
    if len(axes) == 0:
        return keypoints

    # all values in axes are in [1, 2, 3]
    assert np.all(np.array([ax in [1, 2, 3] for ax in axes])), f'{axes} does not contain values from [1, 2, 3]'

    keys = np.asarray(keypoints)

    ndim = keys.shape[1]
    mult = np.ones(ndim, int)
    add = np.zeros(ndim, int)
    for ax in axes:
        mult[ax - 1] = -1
        add[ax - 1] = img_shape[ax - 1] - 1

    keys = keys * mult + add

    return [tuple(k) for k in keys.tolist()]


# Used in rot90_keypoints
def transpose_keypoints(keypoints, ax1, ax2):
    # all values in axes are in [1, 2, 3]
    assert (ax1 in [1, 2, 3]) and (ax2 in [1, 2, 3]), f'[{ax1} {ax2}] does not contain values from [1, 2, 3]'

    axis1 = ax1 - 1
    axis2 = ax2 - 1
    keys = np.asarray(keypoints)
    keys[:, axis1], keys[:, axis2] = keys[:, axis2], keys[:, axis1].copy()

    # Return a list of tuples
    return [tuple(k) for k in keys.tolist()]


def rot90_keypoints(keypoints, factor, axes, img_shape):
    """
    Rotates keypoints by 0, 90, 180, and 270 degrees.
    Returns rotated keypoints and the shape of the rotated image domain.
    """
    img_shape = np.array(img_shape)

    if factor == 1:
        keypoints = flip_keypoints(keypoints, [axes[1]], img_shape)
        keypoints = transpose_keypoints(keypoints, axes[0], axes[1])
        img_shape[axes[0] - 1], img_shape[axes[1] - 1] = img_shape[axes[1] - 1], img_shape[axes[0] - 1]

    elif factor == 2:
        keypoints = flip_keypoints(keypoints, axes, img_shape)

    elif factor == 3:
        keypoints = transpose_keypoints(keypoints, axes[0], axes[1])
        img_shape[axes[0] - 1], img_shape[axes[1] - 1] = img_shape[axes[1] - 1], img_shape[axes[0] - 1]
        keypoints = flip_keypoints(keypoints, [axes[1]], img_shape)

    return keypoints, img_shape


def pad_keypoints(keypoints, pad_size):
    a, b, c, d, e, f = pad_size

    keys = np.asarray(keypoints)
    padding = np.asarray((a, c, e) if keys.shape[1] == 3 else (a, c, e, 0))  # we only need the 'before' pad size

    # Return a list of tuples
    return [tuple(k) for k in (keys + padding).tolist()]


def pad_pixels(img, input_pad_width: TypeSextetInt, border_mode, cval, mask=False):
    # convert the padding argument to appropriate format
    a, b, c, d, e, f = input_pad_width
    pad_width = [(a, b), (c, d), (e, f)]

    # zeroes for channel dimension
    if not mask:
        pad_width = [(0, 0)] + pad_width

    # zeroes for temporal dimension
    if len(img.shape) > len(pad_width):
        pad_width = pad_width + [(0, 0)]

    assert len(img.shape) == len(pad_width)

    # pad and return
    if border_mode == 'constant':
        return np.pad(img, pad_width, border_mode, constant_values=cval)
    if border_mode == 'linear_ramp':
        return np.pad(img, pad_width, border_mode, end_values=cval)
    return np.pad(img, pad_width, border_mode)


# Used in crop()
def get_spatial_shape(array: np.array, mask: bool) -> TypeSpatialShape:
    return np.array(array.shape)[:3] if mask else np.array(array.shape)[1:4]  # mask has no channel dim


# Used in crop()
def get_pad_dims(spatial_shape: TypeSpatialShape, crop_shape: TypeSpatialShape) -> TypeSextetInt:
    pad_dims = [0] * 6
    for i in range(3):  # for each spatial axis
        i_dim, c_dim = spatial_shape[i], crop_shape[i]
        current_pad_dims = (0, 0)
        if i_dim < c_dim:  # if we want larger crop than is the size of the image (in the given axis) --> we must pad:
            pad_size = c_dim - i_dim
            if pad_size % 2 != 0:
                current_pad_dims = (int(pad_size // 2 + 1), int(pad_size // 2))
            else:
                current_pad_dims = (int(pad_size // 2), int(pad_size // 2))

        pad_dims[i * 2:(i + 1) * 2] = current_pad_dims  # store the axis padding tuple (before, after) to pad_dims

    return tuple(pad_dims)


def crop(input_array: np.array,
         crop_shape: TypeSpatialShape,
         crop_position: TypeSpatialShape,
         pad_dims,
         border_mode, cval, mask):

    if crop_shape is None:
        return input_array

    input_spatial_shape = get_spatial_shape(input_array, mask)  # get shape for the spatial dims only

    # if we want larger crop than is the size of the image (in any axis), we must pad the axis
    if np.any(input_spatial_shape < crop_shape):
        warn(f'F.crop(): Input size {input_spatial_shape} smaller than crop size {crop_shape}, pad by {border_mode}.',
             UserWarning)

        # pad
        input_array = pad_pixels(input_array, pad_dims, border_mode, cval, mask)

        # test
        input_spatial_shape = get_spatial_shape(input_array, mask)
        assert np.all(input_spatial_shape >= crop_shape)

    x1, y1, z1 = crop_position
    x2, y2, z2 = np.array(crop_position) + np.array(crop_shape)

    if mask:
        result = input_array[x1:x2, y1:y2, z1:z2]
        assert np.all(result.shape[:3] == crop_shape), f'{result.shape} {crop_shape} {mask} {crop_position}'
    else:
        result = input_array[:, x1:x2, y1:y2, z1:z2]
        assert np.all(result.shape[1:4] == crop_shape)

    return result


def crop_keypoints(keypoints,
                   crop_shape: TypeSpatialShape,
                   crop_position: TypeSpatialShape,
                   pad_dims,
                   keep_all: bool):

    if crop_shape is None:
        return keypoints

    # Get padding information
    px, _, py, _, pz, _ = pad_dims  # we only need the 'before' padding size
    pad = np.asarray((px, py, pz))

    # Compute new keypoint positions
    keys = np.asarray(keypoints)[:, :3] - np.asarray(crop_position) + pad  # ignore the time dimension of keypoints for the computation
    keys = np.concat([keys, np.asarray(keypoints)[:, 3:]], axis=-1)  # add the time dimension to the transformed positions

    # Filter the keypoints
    if not keep_all:
        mask = (keys[:, :3] >= 0) & (keys[:, :3] + .5 < np.asarray(crop_shape))
        keys = keys[np.sum(mask, axis=1) == 3, :]

    # Return a list of tuples
    return [tuple(k) for k in keys.tolist()]


def gaussian_blur(img, input_sigma, border_mode, cval):
    sigma = input_sigma

    # if sigma is of type list, we have different sigma for each channel --> delegate to function gaussian_blur_stack()
    if isinstance(sigma, list):
        if img.shape[0] != len(sigma):
            warn(f'gaussian blur: sigma list length ({len(sigma)}) should equal the number of '
                 f'channels ({img.shape[0]}). Skipping the transformation.', UserWarning)
            return img
        return gaussian_blur_stack(img, sigma, border_mode, cval)

    if isinstance(sigma, (int, float)):
        # replicate sigma for all spatial dimensions and add dummy sigma for C and T dimensions

        sigma = [sigma] * len(img.shape)
        sigma[0] = 0  # channel dim = no blur
        # Checking for time dimension
        if len(img.shape) > 4:
            sigma[-1] = 0  # time dime = no blur
    else:
        # sigma is a tuple

        sigma = (0,) + sigma  # channel dim = no blur

        if len(sigma) < len(img.shape):
            sigma = sigma + (0,)  # time dim without blur
        elif len(sigma) > len(img.shape):  # time dim in sigma [0, sz, sy, sx, st] but not in the image [C, Z, Y, X]
            warn(f'gaussian blur: Too many values for sigma ({sigma[1:]}), only using the first three values.')
            sigma = sigma[:len(img.shape)]

    # check if we have a correct format of sigma
    if len(sigma) != len(img.shape):
        warn(f'gaussian blur: wrong sigma tuple (length does not equal the number of affected dimensions). '
             f'Skipping the transformation.', UserWarning)
        return img

    # compute
    return gaussian_filter(img, sigma=sigma, mode=border_mode, cval=cval)


def gaussian_blur_stack(img, input_sigma, border_mode, cval):
    sigma = (np.asarray(input_sigma).copy()).tolist()

    # simple sigma check
    for channel in sigma:
        if not isinstance(channel, (float, int, tuple)):
            warn(f'gaussian blur: wrong sigma format: the list can only contain elements of type tuple, float or int. '
                 f'Skipping the transformation.', UserWarning)
            return img

    # check if we have a value for each channel
    if len(sigma) != img.shape[0]:
        warn(f'gaussian blur: the number of sigma values ({len(sigma)}) does not equal the number of image channels '
             f'({img.shape[0]}). Skipping the transformation.', UserWarning)
        return img

    image_dim_without_channels = len(img.shape) - 1
    res = np.empty_like(img)
    for i in range(len(sigma)):  # for each channel

        if isinstance(sigma[i], (float, int)):
            # replicate sigma for all spatial dimensions and add dummy sigma for C and T dimensions

            sigma[i] = [sigma[i]] * image_dim_without_channels  # no channel dim
            if image_dim_without_channels > 3:
                sigma[i][-1] = 0  # time dime = no blur

        else:
            # sigma is a tuple

            if len(sigma[i]) < image_dim_without_channels:
                sigma[i] = sigma[i] + (0,)  # time dim without blur
            elif len(sigma[i]) > image_dim_without_channels:  # time dim in sigma [sz, sy, sx, st] but not in the image [(C,) Z, Y, X]
                warn(f'gaussian blur: Too many values for channel sigma ({sigma[i]}), only using the first three values.')
                sigma[i] = sigma[i][:image_dim_without_channels]

        res[i, ...] = gaussian_filter(img[i], sigma=sigma[i], mode=border_mode, cval=cval)  # compute

    return res


def brightness_contrast_adjust(img, alpha=1, beta=0):
    if alpha != 1:
        img *= alpha
    if beta != 0:
        img += beta
    return img


def gamma_transform(img, gamma):
    if np.all(img < 0) or np.all(img > 1):
        warn(f'Gamma transform: image is not in range [0, 1]. Skipping the transformation.', UserWarning)
        return img
    else:
        return np.power(img, gamma)


def histogram_equalization(img, bins):
    for i in range(img.shape[0]):  # for each channel
        img[i] = equalize_hist(img[i], bins)
    return img


def gaussian_noise(img, mean, sigma):
    img = img.astype('float32')
    noise = normal(mean, sigma, img.shape).astype(np.float32)
    return img + noise


def poisson_noise(img):
    return img.astype(np.float32) + poisson(img).astype(np.float32)


def value_to_list(value, length):
    if isinstance(value, (float, int)):
        return [value] * length
    else:
        return list(value)


def correct_length_list(list_to_check, length, value_to_fill=1, list_name='###Default###'):

    if len(list_to_check) < length:
        warn(f'{list_name} have elements {len(list_to_check)}, should be {length} appending {value_to_fill} ' +
             'till length matches', UserWarning)
        for i in range(length - len(list_to_check)):
            list_to_check = list_to_check + [value_to_fill]

    if len(list_to_check) > length:
        warn(f'{list_name} have elements {len(list_to_check)}, should be {length} removing elements from behind ' +
             ' till length matches', UserWarning)
        list_to_check = [list_to_check[i] for i in range(length)]

    return list_to_check


def normalize(img, input_mean, input_std):
    """
    Normalize a multi-channel image to have the desired mean and standard deviation values.
    """

    mean = value_to_list(input_mean, img.shape[0])
    std = value_to_list(input_std, img.shape[0])

    mean = correct_length_list(mean, img.shape[0], value_to_fill=0, list_name='mean')
    std = correct_length_list(std, img.shape[0], value_to_fill=1, list_name='std')

    for i in range(img.shape[0]):  # for each channel
        img[i] = normalize_channel(img[i], mean[i], std[i])

    return img


def normalize_channel(img, mean, std):
    """
    Normalize a single-channel image to have the desired mean and standard deviation values.

    Formula from: https://stats.stackexchange.com/questions/46429/transform-data-to-desired-mean-and-standard-deviation
    """
    return (img - img.mean()) * (std / img.std()) + mean


def normalize_mean_std(img, mean, denominator):
    img -= atleast_kd(mean, k=img.ndim)
    img *= atleast_kd(denominator, k=img.ndim)
    return img
