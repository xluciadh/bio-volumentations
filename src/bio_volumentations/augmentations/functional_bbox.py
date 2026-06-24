# ============================================================================================= #
#  Author:       Jakub Polonský, Lucia Hradecká                                                 #
#  Copyright:    Jakub Polonský                                                                 #
#                Lucia Hradecká     : lucia.d.hradecka@gmail.com                                #
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


from typing import Optional
import itertools
from numpy.typing import NDArray

import numpy as np

from ..biovol_typing import TypeSextetInt, TypeSpatialShape, TypeSpatioTemporalCoordinate, TypeTripletFloat


class BoundingBox:
    """Class that represents bounding boxes as objects

        Internally implemented in the 'voc' format - absolute coordinates
        of the minimal and maximal points of the bounding box.

        Optionally supports a class label.

        Args:
            min_point (NDArray[np.float64], shape (3,)): Minimal point of the bounding box.
                The order of dimensions is (Z Y X).

            max_point (NDArray[np.float64], shape (3,)): Maximal point of the bounding box.
                The order of dimensions is (Z Y X).

            time_point (int): Temporal attribute of the bounding box.

            class_label (str, optional): Optional class label for the bounding box.

                Defaults to ``None``
            bbox_format (str): Format the bounding box was supplied in and will be output back.
                Supported formats: 'voc', 'coco', 'yolo', 'albumentations'.

                Defaults to `'voc'`.
            shape (TypeTripletFloat, optional): Shape of the input image.

                Defaults to ``None``
    """

    def __init__(self,
                 min_point: NDArray[np.float64],
                 max_point: NDArray[np.float64],
                 time_point: int,
                 class_label: Optional[str] = None,
                 bbox_format: str = 'voc',
                 shape: Optional[TypeTripletFloat] = None):
        self.min = min_point
        self.max = max_point

        self.time_point = time_point
        self.bbox_format = bbox_format
        self.class_label = class_label

        self.shape = shape

    def __add__(self, value):
        self.min = self.min + value
        self.max = self.max + value

        return self

    def __mul__(self, value):
        self.min = self.min * value
        self.max = self.max * value

        return self

    def transpose(self, axis1: int, axis2: int) -> None:
        self.min[axis1], self.min[axis2] = self.min[axis2], self.min[axis1]
        self.max[axis1], self.max[axis2] = self.max[axis2], self.max[axis1]

    def crop(self, crop_shape: TypeSpatialShape) -> None:
        self.min = np.clip(self.min, 0, np.asarray(crop_shape) - 1)
        self.max = np.clip(self.max, 0, np.asarray(crop_shape) - 1)

    def get_volume(self) -> float:
        return np.prod(self.max - self.min)

    def is_in_domain_limit(self, domain_limit: TypeSpatialShape) -> bool:
        return (self.max >= (0, 0, 0)).all() and (self.min < domain_limit).all()


def flip_bboxes(bboxes: list[BoundingBox], axes: tuple[int, ...], img_shape: NDArray) -> list[BoundingBox]:
    if len(axes) == 0:
        return bboxes

    # all values in axes are in [1, 2, 3]
    assert np.all(np.array([ax in [1, 2, 3] for ax in axes])), f'{axes} does not contain values from [1, 2, 3]'

    ndim = 3
    mult = np.ones(ndim, int)
    add = np.zeros(ndim, int)
    for ax in axes:
        mult[ax - 1] = -1
        add[ax - 1] = img_shape[ax - 1] - 1

    for bbox in bboxes:
        bbox *= mult
        bbox += add
        bbox.min, bbox.max = np.minimum(bbox.min, bbox.max), np.maximum(bbox.min, bbox.max)

    return bboxes


def transpose_bboxes(bboxes: list[BoundingBox], ax1: int, ax2: int):
    # all values in axes are in [1, 2, 3]
    assert (ax1 in [1, 2, 3]) and (ax2 in [1, 2, 3]), f'[{ax1} {ax2}] does not contain values from [1, 2, 3]'

    axis1 = ax1 - 1
    axis2 = ax2 - 1

    for bbox in bboxes:
        bbox.transpose(axis1, axis2)

    return bboxes


def rot90_bboxes(bboxes: list[BoundingBox], factor: int, axes: tuple[int, ...], img_shape: NDArray):
    if factor == 1:
        bboxes = flip_bboxes(bboxes, (axes[1],), img_shape)
        bboxes = transpose_bboxes(bboxes, axes[0], axes[1])
        img_shape[axes[0] - 1], img_shape[axes[1] - 1] = img_shape[axes[1] - 1], img_shape[axes[0] - 1]

    elif factor == 2:
        bboxes = flip_bboxes(bboxes, axes, img_shape)

    elif factor == 3:
        bboxes = transpose_bboxes(bboxes, axes[0], axes[1])
        img_shape[axes[0] - 1], img_shape[axes[1] - 1] = img_shape[axes[1] - 1], img_shape[axes[0] - 1]
        bboxes = flip_bboxes(bboxes, (axes[1],), img_shape)

    for bbox in bboxes:
        bbox.min, bbox.max = np.minimum(bbox.min, bbox.max), np.maximum(bbox.min, bbox.max)

        # updates the image shape for normalization
        if bbox.shape is not None:
            bbox.shape = img_shape

    return bboxes, img_shape


def resize_bboxes(bboxes: list[BoundingBox],
                  domain_limit: TypeSpatialShape,
                  new_shape: TypeSpatialShape):
    if new_shape is None:
        return bboxes

    assert len(domain_limit) == len(new_shape) == 3

    ratio = np.array(new_shape) / np.array(domain_limit)

    for bbox in bboxes:
        bbox *= ratio

        # updates the image shape for normalization
        if bbox.shape is not None:
            bbox.shape = domain_limit * ratio

    return bboxes


def pad_bboxes(bboxes: list[BoundingBox], pad_size: TypeSextetInt):
    a, b, c, d, e, f = pad_size

    for bbox in bboxes:
        bbox += (a, c, e)

        # updates the image shape for normalization
        if bbox.shape is not None:
            bbox.shape = bbox.shape + (a + b, c + d, e + f)

    return bboxes


def crop_bboxes(bboxes: list[BoundingBox],
                crop_shape: TypeSpatialShape,
                crop_position: TypeSpatialShape,
                pad_dims: TypeSextetInt,
                keep_all: bool,
                min_volume: float | None,
                min_percentage: float | None):

    if crop_shape is None:
        return bboxes

    a, b, c, d, e, f = pad_dims

    res = []
    for bbox in bboxes:
        old_volume = bbox.get_volume()

        bbox += -np.asarray(crop_position) + np.asarray((a, c, e))

        if keep_all:
            res.append(bbox)
            continue

        if not bbox.is_in_domain_limit(crop_shape):
            continue

        bbox.crop(crop_shape)

        if np.any(bbox.min >= bbox.max):
            continue

        # updates the image shape for normalization
        if bbox.shape is not None:
            bbox.shape = crop_shape

        if min_volume is None and min_percentage is None:
            res.append(bbox)

        elif (min_volume is not None and bbox.get_volume() >= min_volume) or \
                (min_percentage is not None and bbox.get_volume() / old_volume >= min_percentage):
            res.append(bbox)

    return res


def affine_bboxes(bboxes: list[BoundingBox],
                  transform,
                  domain_limit: TypeSpatioTemporalCoordinate,
                  keep_all: bool,
                  min_volume: float | None,
                  min_percentage: float | None,
                  degrees: TypeTripletFloat = (0, 0, 0),
                  ):
    """Compute affine transformation of a set of bounding boxes.

    Args:
        bboxes: list of input bounding boxes (spatial coordinates in ZYX format)
        transform: sitk transformation to be applied to the bounding boxes
        domain_limit: limit of the domain (in ZYXT format), there bounding boxes can appear, it is used to define center of transforms
                and to filter out output bounding boxes from the outside of the domain
        keep_all: True to keep also bounding boxes outside the image domain
        min_volume: min absolute bbox volume to keep it
        min_percentage: min relative bbox volume to keep it
        degrees: angles by which the domain is rotated (in XYZ format)

    Returns:
        list: A list of transformed bounding boxes (spatial coordinates in ZYX format).

    """

    transform = transform.GetInverse()

    res = []
    for bbox in bboxes:
        old_volume = bbox.get_volume()

        # apply the transform
        if sum(degrees) == 0:
            # scaling and translation only - it is sufficient to only transform the min and max corners
            bbox.min = np.asarray(transform.TransformPoint(bbox.min.tolist()[::-1])[::-1], float)
            bbox.max = np.asarray(transform.TransformPoint(bbox.max.tolist()[::-1])[::-1], float)

        else:
            # we need to align the bbox after the transformation - we need to transform all corners of the bbox
            # 1. for each axis, determine the two possible values:
            mins = np.minimum(bbox.min, bbox.max)
            maxs = np.maximum(bbox.min, bbox.max)
            # 2. generate all combinations of (z, y, x):
            all_corners_orig = [np.asarray([z, y, x]) for z, y, x in
                                itertools.product([mins[0], maxs[0]], [mins[1], maxs[1]], [mins[2], maxs[2]])]
            # 3. transform all corners:
            all_corners_transformed = [np.asarray(transform.TransformPoint(corner.tolist()[::-1])[::-1], float)
                                       for corner in all_corners_orig]
            # 4. find the new min and max points:
            bbox.min = np.array(all_corners_transformed).min(axis=0)
            bbox.max = np.array(all_corners_transformed).max(axis=0)

        if keep_all:
            res.append(bbox)
            continue

        if not bbox.is_in_domain_limit(domain_limit[:3]):
            continue

        bbox.crop(domain_limit[:3])

        if np.any(bbox.min >= bbox.max):
            continue

        # updates the image shape for normalization
        if bbox.shape is not None:
            bbox.shape = domain_limit[:3]

        if min_volume is None and min_percentage is None:
            res.append(bbox)

        elif (min_volume is not None and bbox.get_volume() >= min_volume) or \
                (min_percentage is not None and bbox.get_volume() / old_volume >= min_percentage):
            res.append(bbox)

    return res
