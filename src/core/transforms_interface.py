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


import random

# DEBUG only flag
VERBOSE = False


class Transform:
    """The base class for transformations.

        Args:
            always_apply (bool, optional): Always apply this transformation.

                Defaults to ``False``.
            p (float, optional): Chance of applying this transformation.

                Defaults to ``0.5``.
    """
    def __init__(self, always_apply=False, p=0.5):
        assert 0 <= p <= 1
        self.p = p
        self.always_apply = always_apply

    def __call__(self, force_apply, targets, **data):
        if force_apply or self.always_apply or random.random() < self.p:
            params = self.get_params(**data)

            if VERBOSE:
                print('RUN', self.__class__.__name__, params)

            for k, v in data.items():
                if k in targets['img_keywords']:
                    data[k] = self.apply(v, **params)
                else:
                    # no transformation
                    pass

        return data

    def get_params(self, **data):
        # Shared parameters for one apply (usually random values).
        return {}

    def apply(self, volume, **params):
        raise NotImplementedError


class DualTransform(Transform):
    """The base class of transformations applied images and also to all target types.

        Targets:
            image, mask, float mask, key points, bounding boxes
    """

    def __call__(self, force_apply, targets, **data):
        if force_apply or self.always_apply or random.random() < self.p:
            params = self.get_params(**data)

            if VERBOSE:
                print('RUN', self.__class__.__name__, params)

            for k, v in data.items():
                if k in targets['img_keywords']:
                    data[k] = self.apply(v, **params)
                elif k in targets['mask_keywords']:
                    data[k] = self.apply_to_mask(v, **params)
                elif k in targets['fmask_keywords']:
                    data[k] = self.apply_to_float_mask(v, **params)
                elif k in targets['keypoint_keywords']:
                    data[k] = self.apply_to_keypoints(v, **params)
                elif k in targets['bbox_keywords']:
                    data[k] = self.apply_to_bboxes(v, **params)
                else:
                    # no transformation
                    pass

        return data

    def apply_to_mask(self, mask, **params):
        # default: use image transformation
        return self.apply(mask, **params)
    
    def apply_to_float_mask(self, float_mask, **params):
        # default: use mask transformation
        return self.apply_to_mask(float_mask, **params)

    def apply_to_keypoints(self, keypoints, keep_all=False, **params):
        # default: no transformation
        return keypoints

    def apply_to_bboxes(self, bboxes, **params):
        # default: no transformation
        return bboxes


class ImageOnlyTransform(Transform):
    """The base class of transformations applied to the `image` target only.

        Targets:
            image
    """
    @property
    def targets(self):
        return {"image": self.apply}

