# ============================================================================================= #
#  Author:       Pavel Iakubovskii, ZFTurbo, ashawkey, Dominik Müller,                          #
#                Samuel Šuľan, Lucia Hradecká, Filip Lux, Jakub Polonský                        #
#  Copyright:    albumentations:    : https://github.com/albumentations-team                    #
#                Pavel Iakubovskii  : https://github.com/qubvel                                 #
#                ZFTurbo            : https://github.com/ZFTurbo                                #
#                ashawkey           : https://github.com/ashawkey                               #
#                Dominik Müller     : https://github.com/muellerdo                              #
#                Lucia Hradecká     : lucia.d.hradecka@gmail.com                                #
#                Filip Lux          : lux.filip@gmail.com                                       #
#                Jakub Polonský                                                                 #
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


from warnings import warn

import numpy as np

from ..random_utils import random
from ..core.transforms_interface import DualTransform
from ..conversion import functional as FCT
from ..augmentations.utils import get_spatial_shape_from_image
from ..augmentations import functional_bbox as FB


class Contiguous(DualTransform):
    """Transform the image data to a contiguous array.

        Args:
            always_apply (bool, optional): Always apply this transformation in composition.

                Defaults to ``True``.
            p (float, optional): Probability of applying this transformation in composition.

                Defaults to ``1``.

        Targets:
            image, mask, float mask
    """

    def __init__(self, always_apply: bool = True, p: float = 1.0):
        super().__init__(always_apply, p)

    def apply(self, image, **params):
        return np.ascontiguousarray(image)

    def apply_to_mask(self, mask, **params):
        return np.ascontiguousarray(mask)

    def __repr__(self):
        return f'Contiguous(always_apply={self.always_apply}, p={self.p})'


class StandardizeDatatype(DualTransform):
    """Change image and float_mask datatype to ``np.float32`` without changing the intensities.
        Change mask datatype to ``np.int32``.

        Args:
            always_apply (bool, optional): Always apply this transformation in composition.

                Defaults to ``True``.
            p (float, optional): Probability of applying this transformation in composition.

                Defaults to ``1``.

        Targets:
            image, mask, float mask
    """

    def __init__(self, always_apply: bool = True, p: float = 1.0):
        super().__init__(always_apply, p)

    def apply(self, image, **params):
        return image.astype(np.float32)

    def apply_to_mask(self, mask, **params):
        return mask.astype(np.int32)

    def apply_to_float_mask(self, mask, **params):
        return mask.astype(np.float32)

    def __repr__(self):
        return f'StandardizeDatatype(always_apply={self.always_apply}, p={self.p})'


class ConversionToFormat(DualTransform):
    """Check the very basic assumptions about the input images.

        Add channel dimension to the 3D images without it. Check that shapes of individual target types are
        consistent (to some extent).

        Args:
            always_apply (bool, optional): Always apply this transformation in composition.

                Defaults to ``True``.
            p (float, optional): Probability of applying this transformation in composition.

                Defaults to ``1``.

        Targets:
            image, mask
    """

    def __init__(self, always_apply: bool = True, p: float = 1):
        super().__init__(always_apply, p)

    def __call__(self, force_apply, targets, **data):
        if force_apply or self.always_apply or random() < self.p:

            # get img/mask/float_mask shapes
            img_shape = []
            mask_shape = []
            float_shape = []
            kpt_lists = []
            for k, v in data.items():
                if k in targets['img_keywords']:
                    img_shape.append(v.shape)
                elif k in targets['mask_keywords']:
                    mask_shape.append(v.shape)
                elif k in targets['fmask_keywords']:
                    float_shape.append(v.shape)
                elif k in targets['keypoint_keywords']:
                    kpt_lists.append(v)

            # check that at least 1 image is present
            if len(img_shape) == 0:
                raise RuntimeError(f'No image-type target is present in the sample!')

            # add channel dim to images if necessary
            for k, v in data.items():
                if k in targets['img_keywords']:
                    if len(v.shape) == 3:
                        warn(f'Adding channel dimension to the image', UserWarning)
                        data[k] = v[None, ...]

            img_shape = []
            for k, v in data.items():
                if k in targets['img_keywords']:
                    img_shape.append(v.shape)

            # check image and mask/float_mask shapes
            if not FCT.check_shapes_equal([s[1:] for s in img_shape] + mask_shape + float_shape):
                raise RuntimeError(f'Image and mask shapes are inconsistent. Their spatial dimensions must match.')

            # check img vs kpts dimensionality
            for kpt_list in kpt_lists:
                if len(img_shape[0]) - 1 != FCT.get_keypoints_dim(kpt_list):
                    raise RuntimeError(f'Image and keypoint dimensions do not match: images must have one more '
                                       f'dimension than keypoints.')

            # convert each keypoints target to a list of tuples
            for k, v in data.items():
                if k in targets['keypoint_keywords']:
                    # v should be a list of tuples
                    data[k] = [tuple(kpt.tolist() if isinstance(kpt, np.ndarray) else kpt) for kpt in v]

        return data

    def apply(self, volume, **params):
        return volume

    def apply_to_mask(self, mask, **params):
        return mask

    def apply_to_float_mask(self, mask, **params):
        return mask

    def __repr__(self):
        return f'ConversionToFormat({self.always_apply}, {self.p})'


class KeypointsFixDatatype(DualTransform):
    """Convert any 2D array-like format of keypoints into a list of tuples containing ints of floats.

        The input can be a 2d numpy array, a list of lists, a list of tuples, etc., and individual coordinate values
        can be of type int, float, or any numpy dtype.

        Args:
            always_apply (bool, optional): Always apply this transformation in composition.

                Defaults to ``True``.
            p (float, optional): Probability of applying this transformation in composition.

                Defaults to ``1``.

        Targets:
            key points
    """

    def __init__(self, always_apply: bool = True, p: float = 1):
        super().__init__(always_apply, p)

    def apply(self, volume, **params):
        return volume

    def apply_to_keypoints(self, keypoints, keep_all=False, **params):
        keypoints_ll = np.array(keypoints).tolist()
        return [tuple(keypoint) for keypoint in keypoints_ll]

    def __repr__(self):
        return f'KeypointsFixDatatype({self.always_apply}, {self.p})'


class NoConversion(DualTransform):
    """An identity transform.

        Args:
            always_apply (bool, optional): Always apply this transformation in composition.

                Defaults to ``True``.
            p (float, optional): Probability of applying this transformation in composition.

                Defaults to ``1``.

        Targets:
            image, mask
    """

    def __init__(self, always_apply: bool = True, p: float = 1):
        super().__init__(always_apply, p)

    def apply(self, volume, **params):
        return volume

    def apply_to_mask(self, mask, **params):
        return mask

    def apply_to_float_mask(self, mask, **params):
        return mask

    def __repr__(self):
        return f'NoConversion({self.always_apply}, {self.p})'


class ConvertToBBoxes(DualTransform):
    """Converts bounding boxes from 'raw data' to objects of BoundingBox class.

        Args:
            bbox_format (str): Format the bounding boxes are supplied in.
                Supported formats: 'voc', 'coco', 'yolo', 'albumentations'.

                Defaults to `'voc'`.
            always_apply (bool, optional): Always apply this transformation in composition.

                Defaults to ``True``.
            p (float, optional): Probability of applying this transformation in composition.

                Defaults to ``1``.

        Targets:
            bounding_box

    """

    def __init__(self, bbox_format: str = 'voc', always_apply: bool = True, p: float = 1.0):
        super().__init__(always_apply, p)
        self.bbox_format = bbox_format

    def apply(self, volume, **params):
        return volume

    def apply_to_bboxes(self, bboxes, **params):
        match (self.bbox_format):
            case 'voc':
                return [FB.BoundingBox(
                    min_point=np.asarray(bbox[0], float),
                    max_point=np.asarray(bbox[1], float),
                    time_point=bbox[2],
                    class_label=bbox[3] if len(bbox) == 4 else None,
                    bbox_format=self.bbox_format)
                    for bbox in bboxes]

            case 'coco':
                return [FB.BoundingBox(
                    min_point=np.asarray(bbox[0]),
                    max_point=np.asarray(bbox[0]) + np.asarray(bbox[1]),
                    time_point=bbox[2],
                    class_label=bbox[3] if len(bbox) == 4 else None,
                    bbox_format=self.bbox_format)
                    for bbox in bboxes]

            case 'albumentations':
                return [FB.BoundingBox(
                    min_point=np.asarray(bbox[0]) * params['shape'],
                    max_point=np.asarray(bbox[1]) * params['shape'],
                    time_point=bbox[2],
                    class_label=bbox[3] if len(bbox) == 4 else None,
                    shape=params['shape'],
                    bbox_format=self.bbox_format)
                    for bbox in bboxes]

            case 'yolo':
                return [FB.BoundingBox(
                    min_point=np.asarray([
                        (bbox[0][0] - (bbox[1][0] / 2)) * params['shape'][0],
                        (bbox[0][1] - (bbox[1][1] / 2)) * params['shape'][1],
                        (bbox[0][2] - (bbox[1][2] / 2)) * params['shape'][2]]),

                    max_point=np.asarray([
                        (bbox[0][0] + (bbox[1][0] / 2)) * params['shape'][0],
                        (bbox[0][1] + (bbox[1][1] / 2)) * params['shape'][1],
                        (bbox[0][2] + (bbox[1][2] / 2)) * params['shape'][2]]),

                    time_point=bbox[2],
                    class_label=bbox[3] if len(bbox) == 4 else None,
                    shape=params['shape'],
                    bbox_format=self.bbox_format)
                    for bbox in bboxes]

    def get_params(self, targets, **data):
        shape = get_spatial_shape_from_image(data, targets)
        return {
            'shape': shape,
        }

    def __repr__(self):
        return f'ConvertToBBoxes({self.bbox_format}, {self.always_apply}, {self.p})'


class ConvertFromBBoxes(DualTransform):
    """Converts bounding boxes back to the 'raw data' format specified previously from the internal BoundingBox classes.

        Args:
            always_apply (bool, optional): Always apply this transformation in composition.

                Defaults to ``True``.
            p (float, optional): Probability of applying this transformation in composition.

                Defaults to ``1``.

        Targets:
            bounding_box

    """

    def __init__(self, always_apply: bool = True, p: float = 1.0):
        super().__init__(always_apply, p)

    def apply(self, volume, **params):
        return volume

    def apply_to_bboxes(self, bboxes, **params):
        return [self.convert_back(bbox) for bbox in bboxes]

    def convert_back(self, bbox):
        res = []

        match bbox.bbox_format:
            case 'voc':
                res = [tuple(bbox.min.tolist()), tuple(bbox.max.tolist())]

            case 'coco':
                res = [tuple(bbox.min.tolist()), tuple((bbox.max - bbox.min).tolist())]

            case 'albumentations':
                res = [tuple((bbox.min / bbox.shape).tolist()), tuple((bbox.max / bbox.shape).tolist())]

            case 'yolo':
                center = (bbox.min + bbox.max) / 2
                res = [tuple((center / bbox.shape).tolist()), tuple(((bbox.max - bbox.min) / bbox.shape).tolist())]

        res.append(bbox.time_point)

        if bbox.class_label is not None:
            res.append(bbox.class_label)

        return res

    def __repr__(self):
        return f'ConvertFromBBoxes({self.always_apply}, {self.p})'
