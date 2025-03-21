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


from warnings import warn

from ..random_utils import random
from ..core.transforms_interface import DualTransform
from ..conversion import functional as FCT


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
            # params = self.get_params(**data)

            img_shape = []
            mask_shape = []
            float_shape = []
            for k, v in data.items():
                if k in targets['img_keywords']:
                    img_shape.append(v.shape) 
                elif k in targets['mask_keywords']:
                    mask_shape.append(v.shape) 
                elif k in targets['fmask_keywords']:
                    float_shape.append(v.shape) 
            
            if FCT.check_dimensions(img_shape):
                warn(f'Input images shapes do not have same length,', UserWarning)
            elif FCT.check_dimensions(mask_shape):
                warn(f'Input masks shapes do not have same length,', UserWarning)
            elif FCT.check_dimensions(float_shape):
                warn(f'Float masks shapes do not have same length,', UserWarning)

            for k, v in data.items():
                if k in targets['img_keywords']:
                    if len(v.shape) == 3:
                        warn(f'Adding channel dimension to the image', UserWarning)
                        data[k] = v[None, ...]

        return data

    def apply(self, volume, **params):
        return volume

    def apply_to_mask(self, mask, **params):
        return mask

    def apply_to_float_mask(self, mask, **params):
        return mask

    def __repr__(self):
        return f'ConversionToFormat({self.always_apply}, {self.p})'


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
