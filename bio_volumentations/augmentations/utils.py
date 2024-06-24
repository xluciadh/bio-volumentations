# ============================================================================================= #
#  Author:       Filip Lux                                                                      #
#  Copyright:    Filip Lux          lux.filip@gmail.com                                         #
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

from typing import Sequence, Union
from ..biovol_typing import TypeSextetFloat, TypeTripletFloat, TypePairFloat, \
    TypeSpatioTemporalCoordinate, TypeSpatialCoordinate, TypePairInt, TypeSextetInt
import numpy as np
import SimpleITK as sitk

from collections.abc import Iterable

DEBUG = False


def parse_limits(input_limit: Union[float, TypePairFloat, TypeTripletFloat, TypeSextetFloat],
                 identity_element: float = 1) -> TypeSextetFloat:

    # input_limit = None
    # returns (ie, ie, ie, ie, ie, ie)
    if input_limit is None:
        return tuple((identity_element, ) * 6)

    # input_limit = x : float
    # returns (ie-x, ie+x, ie-x, ie+x, ie-x, ie+x)
    elif (type(input_limit) is float) or (type(input_limit) is int):
        return tuple((identity_element - input_limit, identity_element + input_limit) * 3)

    # input_limit = (a, b) : TypePairFloat
    # returns (a, b, a, b, a, b)
    elif len(input_limit) == 2:
        a, b = input_limit
        return a, b, a, b, a, b

    # input_limit = (a, b, c) : TypeTripletFloat
    # returns (ie-a, ie+a, ie-b, ie+b, ie-c, ie+c)
    elif len(input_limit) == 3:
        res = []
        for item in input_limit:
            # input_limit = ((a, b), (c, d), (e, f))
            # return (a, b, c, d, e, f)
            if isinstance(item, Iterable):
                for val in item:
                    res.append(float(val))
            # input_limit = (a, b, c)
            # return (ie-a, ie+a, ie-b, ie+b, ie-c, ie+c)
            else:
                res.append(float(identity_element - item))
                res.append(float(identity_element + item))
        return tuple(res)

    # input_limit = (a, b, c, d, e, f)
    # returns (a, b, c, d, e, f)
    elif len(input_limit) == 6:
        return input_limit


def parse_pads(pad_size: Union[int, TypePairInt, TypeSextetInt]) -> TypeSextetInt:

    # pad_size = None
    # returns (0, 0, 0, 0, 0, 0)
    if pad_size is None:
        return 0, 0, 0, 0, 0, 0

    # pad_size = x : int
    # returns (x, x, x, x, x, x)
    elif type(pad_size) is int:
        return tuple((pad_size,) * 6)

    # input_limit = (a, b) : TypePairFloat
    # returns (a, b, a, b, a, b)
    elif len(pad_size) == 2:
        a, b = pad_size
        return a, b, a, b, a, b

    # input_limit = (a, b, c, d, e, f)
    # returns (a, b, c, d, e, f)
    elif len(pad_size) == 6:
        return pad_size


def parse_coefs(coefs: Union[float, TypeTripletFloat],
                identity_element: float = 1) -> TypeTripletFloat:

    # input_limit = None
    # return (ie, ie, ie)
    if coefs is None:
        return tuple((identity_element, ) * 3)
    # return (a, a, a)
    elif isinstance(coefs, (int, float)):
        return coefs, coefs, coefs
    # return (a, b, c)
    elif len(coefs) == 3:
        return coefs


def get_image_center(shape: Union[TypeSpatioTemporalCoordinate, TypeSpatialCoordinate],
                     spacing: TypeTripletFloat = (1., 1., 1.),
                     lps: bool = False) -> TypeTripletFloat:

    center = (np.array(shape)[:3] - 1) / 2

    """ TO REMOVE
    shape = np.array(shape)
    if len(shape) == 3:
        center = (shape - 1) / 2
    else:
        center = (shape[1:4] - 1) / 2

    """
    if lps:
        center = ras_to_lps(center)

    return center * np.array(spacing)


def to_spatio_temporal(shape: tuple) -> TypeSpatioTemporalCoordinate:

    shape = list(shape)
    if len(shape) == 3:
        shape.append(0)

    assert len(shape) == 4
    return tuple(shape)


def to_tuple(param, low=None, bias=None):
    """Convert input argument to min-max tuple

        Args:
            param (scalar, tuple or list of 2+ elements): Input value.
                If value is scalar, return value would be (offset - value, offset + value).
                If value is tuple, return value would be value + offset (broadcasted).
            low:  Second element of tuple can be passed as optional argument
            bias: An offset factor added to each element
    """
    if low is not None and bias is not None:
        raise ValueError("Arguments low and bias are mutually exclusive")

    if param is None:
        return param

    if isinstance(param, (int, float)):
        if low is None:
            param = -param, +param
        else:
            param = (low, param) if low < param else (param, low)
    elif isinstance(param, Sequence):
        param = tuple(param)
    else:
        raise ValueError("Argument param must be either scalar (int, float) or tuple")

    if bias is not None:
        return tuple(bias + x for x in param)

    return tuple(param)


# Simple ITK uses LPS coordinates format
def ras_to_lps(triplet: Sequence[float]):
    return np.array((-1, -1, 1), dtype=float) * np.asarray(triplet)


def np_to_sitk(img: np.array) -> sitk.Image:

    # image in format (c, s1, s2, s3, [t])
    assert len(img.shape) == 5
    channels, w, h, d, frames = img.shape

    sample = np.moveaxis(img, 0, 3)
    sample = sample.reshape((w, h, d, channels * frames))

    # TODO: rather swap axis of parameters than data
    sample = np.swapaxes(sample, 0, 2)

    return sitk.GetImageFromArray(sample)


def sitk_to_np(sitk_img: sitk.Image,
               channels,
               frames=1) -> np.array:

    # shape (d, w, h, c*f)
    img = sitk.GetArrayFromImage(sitk_img)

    if len(img.shape) == 3:
        img = np.expand_dims(img, 3)

    assert channels * frames == img.shape[-1], (f'Number of channels ({channels}) and frames ({frames})'
                                                f'do not correspond to the sitk vector size {img.shape[-1]}')

    # split channels and frames
    w, h, d = img.shape[:3]
    img = img.reshape((w, h, d, channels, frames))

    img = np.swapaxes(img, 0, 2)
    img = np.moveaxis(img, 3, 0)

    # shape (c, w, h, d, f)
    return img


def validate_bbox(new_bbox: tuple,
                  old_bbox: tuple,
                  ratio: float = 0.5) -> bool:

    assert len(new_bbox) == len(old_bbox)

    old_size = get_bbox_size(old_bbox)
    new_size = get_bbox_size(new_bbox)

    return old_size / new_size >= ratio


def get_bbox_size(bbox: tuple) -> float:

    assert len(bbox) % 2 == 0
    dims = np.reshape(np.array(bbox), (-1, 2))

    volume = 1.
    for v_min, v_max in dims:

        assert v_max >= v_min, f'The definition of bbox is invalid {bbox}.'
        volume *= v_max - v_min

    return volume


def get_spatio_temporal_domain_limit(sample: dict) -> TypeSpatioTemporalCoordinate:
    """
    Returns vector of spatio-temporal coordinates of length 4.
    The vector limits a domain of the image.

    Args:
        sample: dictionary

    Returns:

    """

    shape = list(sample['image'].shape)
    if len(shape) == 3:
        limit = shape + [1]
    elif len(shape) == 4:
        limit = shape[1:] + [1]
    elif len(shape) == 5:
        limit = shape[1:5]

    assert len(limit) == 4

    return tuple(limit)


def is_included(shape: Union[TypeSpatialCoordinate, TypeSpatioTemporalCoordinate], coo):

    coo_arr = np.array(coo) + 0.5
    shape_arr = np.array(shape[:3])

    assert len(shape_arr) == len(coo_arr), f'shape: {shape_arr} coo: {coo_arr}'
    res = all(coo_arr >= 0) and (coo_arr < shape_arr).all()

    if DEBUG:
        print('IS INCLUDED', shape, coo, res)

    return res


