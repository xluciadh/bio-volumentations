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
from functools import wraps
import skimage.transform as skt
from skimage.exposure import equalize_hist
from scipy.ndimage import zoom, gaussian_filter
from warnings import warn

from ..biovol_typing import TypeTripletFloat
from .spatial_funcional import get_affine_transform, apply_sitk_transform


MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}

# SITK interpolations
SITK_interpolation = {
    0: 'sitkNearestNeighbor',
    1: 'sitkLinear',
    2: 'sitkBSpline',
    3: 'sitkGaussian'
}

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
** Be sure to change volume and mask data type to float !!! **  (already done by Float() in compose)

But for parameters use primarily ints.
"""


def preserve_shape(func):
    """
    Preserve shape of the image
    """

    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape
        result = func(img, *args, **kwargs)
        result = result.reshape(shape)
        return result

    return wrapped_function


def get_center_crop_coords(img_shape, crop_shape):
    froms = (img_shape - crop_shape) // 2
    tos = froms + crop_shape
    return froms, tos


# Too similar to the random_crop. Could be made into one function
def center_crop(img, input_crop_shape, border_mode, cval, mask):
    for i in input_crop_shape:
        if i < 0:
            warn(f'CenterCrop(): shape {input_crop_shape} contains zero or negative number, continuing'
                 f'without CenterCrop.', UserWarning)
            return img
    if not mask:
        crop_shape = np.insert(input_crop_shape, 0, img.shape[0])
    else:
        crop_shape = input_crop_shape
    img_shape = np.array(img.shape)

    # Adding last dimension, if there is one less in the crop_shape
    if len(img_shape) == len(crop_shape) + 1:
        crop_shape = np.append(crop_shape, img_shape[-1])
    if np.any(img_shape < crop_shape):
        warn(f'CenterCrop(): image size {img_shape} smaller than crop size {crop_shape}, pad by {border_mode}.', UserWarning)
        img = pad(img, img_shape, crop_shape, border_mode , cval)
        img_shape = np.array(img.shape)
   
    from_indices, to_indices = get_center_crop_coords(img_shape, crop_shape)
    return crop_from_to(img, from_indices, to_indices)


def pad(img, img_shape, crop_shape, border_mode, cval):
    axes_to_pad = np.where(img_shape < crop_shape)[0]
    pad_width = []
    for i in range(len(img_shape)):
        if i in axes_to_pad:
            how_many_to_pad = crop_shape[i] - img_shape[i]
            if how_many_to_pad % 2:
                pad_width.append((int(how_many_to_pad // 2 + 1), int(how_many_to_pad // 2)))
            else:
                pad_width.append((int(how_many_to_pad // 2), int(how_many_to_pad // 2)))
        else:
            pad_width.append((0, 0))
    if border_mode == "constant":
        return np.pad(img, pad_width, border_mode, constant_values=cval)
    if border_mode == "linear_ramp":
        return np.pad(img, pad_width, border_mode, end_values=cval)
    return np.pad(img, pad_width, border_mode)


def pad_pixels(img, input_pad_width, border_mode, cval, mask=False):
    img_shape = img.shape
    if not mask:
        img_shape = img_shape[1:]

    if isinstance(input_pad_width, (int, tuple)):
        pad_width = input_pad_width
    else:
        pad_width = input_pad_width.copy()

    if isinstance(pad_width, int):
        # padding only spatial dimensions
        pad_width = [(pad_width, pad_width) if i < 3 else (0, 0) for i in range(len(img_shape))]
    elif isinstance(pad_width, tuple):
        if len(pad_width) > 2:
            warn(f'Pad(): tuple for pad_size {pad_width} have more than 2 elements, ignoring elements after the ' +
                 f'second one.', UserWarning)

        pad_width = [(pad_width[0], pad_width[1]) if i < 3 else (0, 0) for i in range(len(img_shape))]

    else:
        # Making tuples out of single numbers
        for i in range(len(pad_width)):
            if isinstance(pad_width[i], int):
                pad_width[i] = (pad_width[i], pad_width[i])
        # Padding with zeroes
        if len(pad_width) < len(img_shape):
            pad_width = pad_width + [(0, 0)] * (len(img_shape) - len(pad_width))

    # zeroes for channel dimension
    if not mask:
        pad_width = [(0, 0)] + pad_width
    
    if border_mode == "constant":
        return np.pad(img, pad_width, border_mode, constant_values=cval)
    if border_mode == "linear_ramp":
        return np.pad(img, pad_width, border_mode, end_values=cval)
    return np.pad(img, pad_width, border_mode)


def crop_from_to(img, froms, tos):
    if len(froms) == 3:
        c1, x1, y1 = froms
        c2, x2, y2 = tos
        return img[c1:c2, x1:x2, y1:y2]
    if len(froms) == 4:
        c1, x1, y1, z1 = froms
        c2, x2, y2, z2 = tos
        return img[c1:c2, x1:x2, y1:y2, z1:z2]
    
    if len(froms) == 5:
        c1, x1, y1, z1, t1 = froms
        c2, x2, y2, z2, t2 = tos
        return img[c1:c2, x1:x2, y1:y2, z1:z2, t1:t2]


def get_random_crop_coords(img_shape, crop_shape, crop_start):
    froms = ((img_shape - crop_shape) * crop_start).astype(int)
    tos = froms + crop_shape
    return froms, tos


# Too similar to the center_crop. Could be made into one function
def random_crop(img, input_crop_shape, input_crop_start, border_mode, cval, mask):
    if not mask:
        crop_shape = np.insert(input_crop_shape, 0, img.shape[0])
        crop_start = np.insert(input_crop_start, 0, 0)
    else:
        crop_shape = input_crop_shape
        crop_start = input_crop_start
    img_shape = np.array(img.shape)

    # Adding last dimension, if there is one less in the crop_shape
    if len(img_shape) == len(crop_shape) + 1:
        crop_shape = np.append(crop_shape, img_shape[-1])
        crop_start = np.append(crop_start, 0)
    if np.any(img_shape < crop_shape):
        warn(
            f'BioVolumentations transformation RandomCrop():' +
            f' image size {img_shape} smaller than crop size {crop_shape}, pad by {border_mode}.',
            UserWarning)
        img = pad(img, img_shape, crop_shape, border_mode,cval)
        img_shape = np.array(img.shape)
    
    froms, tos = get_random_crop_coords(img_shape, crop_shape, crop_start)
    return crop_from_to(img, froms, tos)


def normalize_mean_std(img, mean, denominator):
    if len(mean.shape) == 0:
        mean = mean[..., None]
    if len(denominator.shape) == 0:
        denominator = denominator[..., None]
    new_axis = [i + 1 for i in range(len(img.shape) - 1)]
    img -= np.expand_dims(mean, axis=new_axis)
    img *= np.expand_dims(denominator, axis=new_axis)
    return img


# formula taken from
# https://stats.stackexchange.com/questions/46429/transform-data-to-desired-mean-and-standard-deviation
def normalize_channel(img, mean, std):
    return (img - img.mean()) * (std / img.std()) + mean


def value_to_list(value, length):
    if isinstance(value, (float, int)):
        return [value for _ in range(length)]
    else: 
        return value


def correct_length_list(list_to_check, length, value_to_fill=1, list_name="###Default###"):
    if len(list_to_check) < length:
        warn(f"{list_name} have elements {len(list_to_check)}, should be {length} appending {value_to_fill} " +
             "till length matches", UserWarning)
        for i in range(length - len(list_to_check)):
            list_to_check = list_to_check + [value_to_fill]
    if len(list_to_check) > length:
        warn(f"{list_name} have elements {len(list_to_check)}, should be {length} removing elements from behind " +
             " till length matches", UserWarning)
        list_to_check = [list_to_check[i] for i in range(length)]
    return list_to_check


def normalize(img, input_mean, input_std):
    
    mean = value_to_list(input_mean, img.shape[0])
    std = value_to_list(input_std, img.shape[0])

    mean = correct_length_list(mean, img.shape[0], value_to_fill=0, list_name="mean")
    std = correct_length_list(std, img.shape[0], value_to_fill=1, list_name="std")

    for i in range(img.shape[0]):
        img[i] = normalize_channel(img[i], mean[i], std[i])
    return img


def gaussian_noise(img, mean, sigma):
    img = img.astype("float32")
    noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    return img + noise


def poisson_noise(img, intensity):
    img = img.astype("float32")
    noise = np.random.poisson(img).astype(np.float32) * intensity
    return img + noise


# TODO parameter
# Anti-aliasing - gaussian filter to smooth. using automatically when downsampling, except when integer
# and interpolation is 0. (so mask)
# float mask - how, for now no gaussian filter.
def resize(img, input_new_shape, interpolation=1, border_mode='reflect', cval=0, mask=False,
           anti_aliasing_downsample=True):
    new_shape = input_new_shape

    # Zero or negative check
    for dimension in new_shape:
        if dimension <= 0:
            warn(f"Resize(): shape: {new_shape} contains zero or negative number, continuing without Resize.",
                 UserWarning)
            return img

    # shape check
    if mask:
        # too many or few dimensions of new_shape
        if len(new_shape) < len(img.shape) - 1 or len(new_shape) > len(img.shape):
            warn(f"Resize(): wrong parameter shape:  {new_shape}," +
                 f"expecting something with dimensions of {img.shape } or {img.shape[0:-1] }, " +
                 "continuing without resizing ", UserWarning)
            return img
        # Adding time dimension
        elif len(new_shape) == len(img.shape) - 1:
            new_shape = np.append(new_shape, img.shape[-1])
    else:
        if len(new_shape) < len(img.shape[1:]) - 1 or len(new_shape) > len(img.shape[1:]):
            warn(f"Resize(): wrong dimensions of shape:  {new_shape}," +
                 f"expecting something with dimensions of {img.shape[1:] } or {img.shape[1:-1] }, continuing " +
                 "without resizing ", UserWarning)
            return img
        # adding time dimension
        elif len(new_shape) == len(img.shape[1:]) - 1:
            new_shape = np.append(new_shape, img.shape[-1])

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
    
    if anti_aliasing_downsample and np.any(np.array(img.shape[1:]) < np.array(new_shape)):
        anti_aliasing = True
    
    data = []
    for i in range(img.shape[0]):
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


# TODO compare with skt.rescale, new version got channel_axis
def scale(img, input_scale_factor, interpolation=0, border_mode='reflect', cval=0, mask=True):
    scale_factor = input_scale_factor
    # check for zero or negative numbers
    if isinstance(scale_factor, (int, float)):
        if scale_factor <= 0:
            warn(f"RandomScale()/Scale(): scale_factor: {len(scale_factor)} is zero or negative number" +
                 f" continuing without scaling ", UserWarning)
            return img 
    else:
        for dimension in scale_factor:
            if dimension <= 0:
                warn(f"RandomScale()/Scale(): scale_factor: {len(scale_factor)} contains zero or negative number " +
                     "continuing without scaling ", UserWarning)
                return img 

    img_shape = img.shape
    if scale_factor is None:
        return img
    if isinstance(scale_factor, (list, tuple)):
        scale_factor = np.array(scale_factor)
        if not mask:
            img_shape = img_shape[1:]
        # TODO, maybe user wants to add shape for only spatial dimensions
        if len(img_shape) != len(scale_factor) and len(img_shape) - 1 != len(scale_factor):
            warn(f"RandomScale()/Scale(): Wrong dimension of scaling factor list:  {len(scale_factor)}," +
                 f"expecting {len(img_shape)} or {len(img_shape[:-1]) }, continuing without scaling ", UserWarning)
            return img
        elif len(img_shape) - 1 == len(scale_factor):
            scale_factor = np.append(scale_factor, 1)
    else:
        scale_factor = [scale_factor for _ in range(len(img_shape) - 1)]
        if mask:
            scale_factor.append(scale_factor[0])
        # Not scaling time dimensions
        if len(scale_factor) == 4:
            scale_factor[-1] = 1
    if mask:
        return zoom(img, scale_factor, order=interpolation, mode=border_mode, cval=cval)
    
    data = []
    for i in range(img.shape[0]):
        subimg = img[i].copy()
        d0 = zoom(subimg, scale_factor, order=interpolation, mode=border_mode, cval=cval)
        data.append(d0.copy())
    new_img = np.stack(data, axis=0)
    
    return new_img


'''
#TODO maybe add parameter for order of rotations
#LIMIT dimensions
def affine_transform(img, input_x_angle, input_y_angle, input_z_angle, translantion, interpolation = 1, border_mode = 'constant',
                  value = 0, input_scaling_coef = None, scale_back = True,  mask = False ):
    
    if mask:
        img = img[np.newaxis, :]
    x_angle, y_angle, z_angle = [np.pi * i / 180 for i in [input_x_angle, input_y_angle, input_z_angle]]
    if not(input_scaling_coef is None):
        scaling_coef = np.array(input_scaling_coef)
        #no scaling on the channels if the scaling_coef is in wrong format
        if(len(scaling_coef) != 3):
            warn(f"Rotate transform: Wrong dimension of scaling coeficient list:  {len(scaling_coef)}, expecting {3}, continuing without scaling ", UserWarning)
            inverse_affine_matrix =  np.linalg.inv(rotation_matrix_calculation(len(img.shape),x_angle,y_angle,z_angle ))
        else:
            scaling_coef = np.insert(scaling_coef, 0, 1 )
            if len(scaling_coef) < len(img.shape):
                scaling_coef = np.append(scaling_coef, 1 )
            inverse_scaling_matrix =  np.diag([ 1/i  for i in scaling_coef])
            inverse_rotation_matrix =  np.linalg.inv(rotation_matrix_calculation(len(img.shape),x_angle,y_angle,z_angle ))
            inverse_affine_matrix = inverse_scaling_matrix @ inverse_rotation_matrix
            if scale_back:
                inverse_scale_back_matrix = np.diag([ i  for i in scaling_coef])
                inverse_affine_matrix = inverse_affine_matrix @ inverse_scale_back_matrix

    else:
        inverse_affine_matrix =  np.linalg.inv(rotation_matrix_calculation(len(img.shape),x_angle,y_angle,z_angle ))
    c_in=0.5*np.array(img.shape)
    offset=c_in-inverse_affine_matrix.dot(c_in)
    if not(translantion is None):
        if len(translantion) > len(img.shape) - 1:
            warn(f"Rotate transform(): translation list has wrong length {len(translantion)}, expected {len(img.shape) - 1}", UserWarning)
        else:
            for i in range(len(translantion)):
                offset[i + 1] -= translantion[i]
    img = sci.affine_transform(img, inverse_affine_matrix, offset, order=interpolation, mode=border_mode, cval= value)
    
    if mask:
        img = img[0]
    return img
'''


def affine(img: np.array,
           degrees: TypeTripletFloat = (0, 0, 0),
           scales: TypeTripletFloat = (1, 1, 1),
           translation: TypeTripletFloat = (0, 0, 0),
           interpolation: int = 1,
           border_mode: str = 'constant',
           value: float = 0,
           spacing: TypeTripletFloat = (1, 1, 1)):
    """
    img (np.array) : format (channel, ax1, ax2, ax3, [time])
    """

    transform = get_affine_transform(img,
                                     scales=scales,
                                     degrees=degrees,
                                     translation=translation,
                                     spacing=spacing)

    return apply_sitk_transform(img,
                                sitk_transform=transform,
                                interpolation=SITK_interpolation[interpolation],
                                default_value=value,
                                spacing=spacing)


def rotation_matrix_calculation(dim, x_angle, y_angle, z_angle):
    rot_matrix = np.identity(dim).astype(np.float32)
    rot_matrix = rot_matrix @ rot_x(x_angle, dim)
    rot_matrix = rot_matrix @ rot_y(y_angle, dim)
    rot_matrix = rot_matrix @ rot_z(z_angle, dim)
    return rot_matrix


def rot_x(angle, dim):
    if dim == 4:
        rotation_x = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],  
                               [0, 0, np.cos(angle), -np.sin(angle)],
                               [0, 0, np.sin(angle), np.cos(angle)]])
    if dim == 5:
        rotation_x = np.array([[1, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0],  
                               [0, 0, np.cos(angle), -np.sin(angle), 0],
                               [0, 0, np.sin(angle), np.cos(angle), 0],
                               [0, 0, 0, 0, 1]])
    
    return rotation_x


def rot_y(angle, dim):
    if dim == 4:
        rotation_y = np.array([[1, 0, 0, 0],
                               [0, np.cos(angle), 0, np.sin(angle)],
                               [0, 0, 1, 0],  
                               [0, -np.sin(angle), 0, np.cos(angle)]])
    if dim == 5:
        rotation_y = np.array([[1, 0, 0, 0, 0],
                               [0, np.cos(angle), 0, np.sin(angle), 0],
                               [0, 0, 1, 0, 0],  
                               [0, -np.sin(angle), 0, np.cos(angle), 0],
                               [0, 0, 0, 0, 1]])
    
    return rotation_y


def rot_z(angle, dim):
    if dim == 4:
        rotation_z = np.array([[1, 0, 0, 0],
                               [0, np.cos(angle), -np.sin(angle), 0],
                               [0, np.sin(angle), np.cos(angle), 0],
                               [0, 0, 0, 1]])
    if dim == 5:
        rotation_z = np.array([[1, 0, 0, 0, 0],
                               [0, np.cos(angle), -np.sin(angle), 0, 0],
                               [0, np.sin(angle), np.cos(angle), 0, 0],
                               [0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 1]])
    
    return rotation_z


# TODO clipped tag may be important for types other that float32, but tags are from fork and not tested
# @clipped
def brightness_contrast_adjust(img, alpha=1, beta=0):
    if alpha != 1:
        img *= alpha
    if beta != 0:
        img += beta
    return img


def histogram_equalization(img, bins):
    for i in range(img.shape[0]):
        img[i] = equalize_hist(img[i], bins)
    return img


def gaussian_blur(img, input_sigma, border_mode, cval):
    sigma = input_sigma
    if isinstance(sigma, list):
        if img.shape[0] != len(sigma):
            warn(f'GaussianBlur(): wrong list size {len(sigma)}, expecting same as number of dimensions {img.shape[0]}. Ignoring', UserWarning)
            return img
        return gaussian_blur_stack(img, sigma, border_mode, cval)

    if isinstance(sigma, (int, float)):
        sigma = np.repeat(sigma, len(img.shape))
        sigma[0] = 0
        # Checking for time dimension
        if len(img.shape) > 4:
            sigma[-1] = 0
    else:
        # TODO what to expect in the input.
        if len(sigma) == len(img.shape) - 2:
            sigma = np.append(sigma, 0) 
        if len(sigma) == len(img.shape) - 1: 
            sigma = np.insert(sigma, 0, 0)
    # TODO better warning
    if len(sigma) != len(img.shape):
        warn(f'GaussianBlur(): wrong sigma tuple, ignoring', UserWarning)
        return img
    return gaussian_filter(img, sigma=sigma, mode=border_mode, cval=cval)
    

def gaussian_blur_stack(img, input_sigma, border_mode, cval):
    sigma = list(np.asarray(input_sigma).copy())
    # simple sigma check
    for channel in sigma:
        if not isinstance(channel, (float, int, tuple)):
            warn(f'GaussianBlur(): wrong sigma format, Inside list can be only tuple,float or int. Ignoring',
                 UserWarning)
            return img
    
    # TODO try different techniques for better optimalization.
    for i in range(len(sigma)):
        if isinstance(sigma[i], (float, int)):
            sigma[i] = np.repeat(sigma[i], len(img.shape) - 1)
            if len(sigma[i]) >= 4:
                sigma[i][-1] = 0
        else:
            if len(sigma[i]) == len(img.shape) - 2:
                sigma[i] = np.append(sigma[i], 0)
        img[i] = gaussian_filter(img[i], sigma=sigma[i], mode=border_mode, cval=cval)
    return img


def gamma_transform(img, gamma):
    if np.all(img < 0) or np.all(img > 1) :
        warn(f"Gamma transform: image is not in range [0,1]. continuing without transform", UserWarning)
        return img
    else:
        return np.power(img, gamma)

