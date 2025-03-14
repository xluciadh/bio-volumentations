# ============================================================================================= #
#  Author:       Filip Lux, Lucia Hradecká                                                      #
#  Copyright:    Filip Lux          lux.filip@gmail.com                                         #
#                Lucia Hradecká     lucia.d.hradecka@gmail.com                                  #
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

from typing import Union, Optional
from collections.abc import Iterable
import numpy as np

from ..biovol_typing import TypeSextetFloat, TypeTripletFloat, TypePairFloat, \
    TypeSpatioTemporalCoordinate, TypeSpatialCoordinate, TypePairInt, TypeSextetInt
from ..random_utils import uniform


DEBUG = False


def get_nonchannel_axes(array):
    """Return the non-channel axis indices for a given image.
    """
    return tuple(range(1, array.ndim))


def atleast_kd(array, k):
    """Add singleton dimensions to the input array s.t. the new shape is at least k-dimensional.
    """
    array = np.asarray(array)
    new_shape = array.shape + (1,) * (k - array.ndim)
    return array.reshape(new_shape)


def get_sigma_axiswise(min_sigma, max_sigma):
    """Randomly choose a single sigma for all axes and channels (if max_sigma is int or float)
    or a sigma for each axis (except the channel axis).
    """

    sigma = uniform(min_sigma, max_sigma)

    if isinstance(max_sigma, tuple):
        # If tuple on input, we must return a tuple (not np.ndarray)
        sigma = tuple(sigma)

    return sigma


def get_spatial_shape_from_image(data, targets):
    # Image is always [C, D, H, W] or [C, D, H, W, T]
    return np.array(data[get_first_img_keyword(targets)].shape[1:4])


def parse_limits(input_limit: Union[float, TypePairFloat, TypeTripletFloat, TypeSextetFloat],
                 scale: bool = False) -> TypeSextetFloat:
    """Parse the limits of affine transformation: rotation, scaling, or translation.
    
    Args:
        input_limit: transformation limits (type None, float, tuple of 2 floats, tuple of 3 floats, tuple of 6 floats)
        scale: a flag (True if computing limits for scaling, False otherwise)

    Returns:
        A tuple of 6 floats representing the limits for all 3 spatial dimensions.

        input_limit = None --> return (0., 0., 0., 0., 0., 0.)

        input_limit = x : float --> return (1/x, x, 1/x, x, 1/x, x) if scale, else (-x, +x, -x, +x, -x, +x)

        input_limit = (a, b) : TypePairFloat --> return (a, b, a, b, a, b)

        input_limit = (a, b, c) : TypeTripletFloat --> return (1/a, a, 1/b, b, 1/c, c) if scale, else (-a, +a, -b, +b, -c, +c)

        input_limit = ((a, b), (c, d), (e, f)) : TypeTripletFloat --> return (a, b, c, d, e, f)

        input_limit = (a, b, c, d, e, f) : TypeSextetFloat --> return (a, b, c, d, e, f)
    """

    # input_limit = x : float --> return (1/x, x, 1/x, x, 1/x, x) if scale, else (-x, +x, -x, +x, -x, +x)
    if isinstance(input_limit, float) or isinstance(input_limit, int):
        limit_range = parse_helper_affine_limits_1d(input_limit, scale=scale)  # get (1/x, x) or (-x, +x)
        return limit_range * 3  # copy the tuple for each spatial axis

    # input_limit : TypeTripletFloat
    #    if   input_limit = ((a, b), (c, d), (e, f)) --> return (a, b, c, d, e, f)
    #    elif input_limit = (a, b, c) --> return (-a, +a, -b, +b, -c, +c)
    #                                     if scale, return (1/a, a, 1/b, b, 1/c, c)
    if len(input_limit) == 3:
        res = []
        for item in input_limit:  # for each spatial axis
            if isinstance(item, Iterable):
                # we already have a tuple -> add it to the result
                res.extend(item)
            else:
                # we need to create a tuple
                limit_range = parse_helper_affine_limits_1d(item, scale=scale)  # get (1/x, x) or (-x, +x)
                res.append(limit_range[0])
                res.append(limit_range[1])
        return tuple(res)
        
    return parse_helper_sextet_common_cases(input_limit, return_float=True)


def parse_helper_affine_limits_1d(input_limit: float, scale: bool) -> tuple:
    """Create a 2-tuple of transformation limits for a single spatial axis.
    
    Returns:
        (1/x, x) if scale=True, (-x, +x) otherwise
    """
    return tuple(sorted([input_limit, 1 / input_limit])) if scale else (-input_limit, input_limit)


def parse_pads(pad_size: Union[int, TypePairInt, TypeSextetInt]) -> TypeSextetInt:
    """Parse the padding argument.

    Args:
        pad_size: padding size (type None, int, tuple of 2 ints, tuple of 6 ints)

    Returns:
        A tuple of 6 ints representing padding for all 3 spatial dimensions.

        pad_size = None --> return (0, 0, 0, 0, 0, 0)

        pad_size = x : int --> return (x, x, x, x, x, x)

        input_limit = (a, b) : TypePairInt --> return (a, b, a, b, a, b)

        input_limit = (a, b, c, d, e, f) --> return (a, b, c, d, e, f)
    """

    if isinstance(pad_size, int):
        return tuple((pad_size,) * 6)

    return parse_helper_sextet_common_cases(pad_size, return_float=False)


def parse_helper_sextet_common_cases(arg: Optional[tuple], return_float=False):
    """Parse the arguments of geometric transformations in common cases when type(arg) is None, 2-tuple, or 6-tuple.
    """

    if arg is None:
        elem = 0. if return_float else 0
        return (elem,) * 6

    elif len(arg) == 2:
        return arg * 3

    elif len(arg) == 6:
        return arg


def parse_coefs(coefs: Union[float, tuple], identity_element: float = 1, dim4: bool = False) -> tuple:
    """Parse the coefficients of affine transformation: rotation, scaling, or translation.

    Args:
        coefs: transformation coefficients
        identity_element: identity element (e.g. 1 for scaling, 0 for translation)
        dim4: a flag (True if time-lapse data, False otherwise)

    Returns:
        A tuple of 3 floats representing the transformation parameters for all 3 spatial dimensions.
    """

    # input_limit = None --> return (ie, ie, ie)
    if coefs is None:
        return (identity_element,) * 3

    # return (a, a, a)
    elif isinstance(coefs, (int, float)):
        return (coefs,) * 3

    # return (a, b, c) for 3D data or (a, b, c, d) for time-lapse (4D) data
    elif (len(coefs) == 3) or (dim4 and len(coefs) == 4):
        return coefs


def get_first_img_keyword(targets: dict = None):
    """Get the first 'image'-type keyword from the targets dictionary.
    """

    if (targets is not None) and isinstance(targets, dict):
        return targets.get('img_keywords')[0]
    return 'image'  # <-- best effort, if we don't have concrete naming in the `targets` dict


def get_spatio_temporal_domain_limit(sample: dict, targets: dict = None) -> TypeSpatioTemporalCoordinate:
    """Return a vector of spatio-temporal coordinates of length 4.

    The vector limits the domain of the image.

    Args:
        sample: dictionary with data
        targets: dictionary with targets
    """

    shape = list(sample[get_first_img_keyword(targets)].shape)

    if len(shape) == 3:
        # 3D image without channels and the time axis
        limit = shape + [1]

    elif len(shape) == 4:
        # 3D image with channels, without the time axis
        limit = shape[1:] + [1]

    elif len(shape) == 5:
        # 3D image with channels and the time axis
        limit = shape[1:5]

    assert len(limit) == 4
    return tuple(limit)


def to_spatio_temporal(shape: tuple) -> TypeSpatioTemporalCoordinate:
    """Return spatio-temporal shape given the input shape (without the channel dimension).
    """

    shape = list(shape)
    if len(shape) == 3:
        shape.append(0)

    assert len(shape) == 4
    return tuple(shape)


def to_tuple(param: Union[int, float, Iterable]):
    """Convert input argument to min-max tuple

    Args:
        param (scalar or Iterable): Input value.
            If scalar, the return value is (-value, +value). Otherwise, convert the Iterable to tuple.
    """
    if param is None:
        return param
    if isinstance(param, (int, float)):
        return -param, +param
    return tuple(param)


def is_included(shape: Union[TypeSpatialCoordinate, TypeSpatioTemporalCoordinate], coo):

    coo_arr = np.array(coo) + 0.5
    shape_arr = np.array(shape[:3])  # ignore the time dimension

    assert len(shape_arr) == len(coo_arr), f'shape: {shape_arr} coo: {coo_arr}'
    res = all(coo_arr >= 0) and (coo_arr < shape_arr).all()

    if DEBUG:
        print('IS INCLUDED', shape, coo, res)

    return res


def validate_bbox(new_bbox: tuple, old_bbox: tuple, ratio: float = 0.5) -> bool:

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




