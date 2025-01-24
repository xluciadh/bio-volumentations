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
from ..augmentations import transforms as T
from ..conversion import transforms as CT


class Compose:
    """Compose a list of transformations into a callable transformation pipeline.

    **It is strongly recommended to use** ``Compose`` **to define and use the transformation pipeline.**

    In addition, basic input image checks and conversions are performed. Optionally, datatype conversion
    (e.g. from ``numpy.ndarray`` to ``torch.Tensor``) is performed.

    Args:
        transforms (List[Transform]): A list of transforms (objects of type ``Transform``).

        p (float, optional): The chance of applying the whole pipeline.

            Defaults to ``1``.

        img_keywords (Tuple[str], optional): List of `image` target names.

            Defaults to ``('image',)``.

        mask_keywords (Tuple[str], optional): List of `mask` target names.

            Defaults to ``('mask',)``.

        fmask_keywords (Tuple[str], optional): List of `float mask` target names.

            Defaults to ``('float_mask',)``.

        keypoints_keywords (Tuple[str], optional): List of `key points` target names.

            Defaults to ``('keypoints',)``.

        bboxes_keywords (Tuple[str], optional): List of `bounding boxes` target names.

            Defaults to ``('bboxes',)``.

        value_keywords (Tuple[str], optional): List of `value` target names.

            Defaults to ``('value',)``.

        conversion (Transform | None, optional): Image datatype conversion transform, applied after the transformations.

            Defaults to ``None``.
    """
    def __init__(self,
                 transforms, p=1.0,
                 img_keywords=('image',),
                 mask_keywords=('mask',),
                 fmask_keywords=('float_mask',),
                 keypoints_keywords=('keypoints',),
                 bboxes_keywords=('bboxes',),
                 value_keywords=('value',),
                 conversion=None):

        assert 0 <= p <= 1

        self.transforms = ([T.StandardizeDatatype(always_apply=True),
                            CT.ConversionToFormat(always_apply=True)] +
                           transforms +
                           [T.Contiguous(always_apply=True),
                            CT.NoConversion() if conversion is None else conversion])
        self.p = p
        self.targets = {'img_keywords': img_keywords,
                        'mask_keywords': mask_keywords,
                        'fmask_keywords': fmask_keywords,
                        'keypoint_keywords': keypoints_keywords,
                        'bbox_keywords': bboxes_keywords,
                        'value_keywords': value_keywords}

    def get_always_apply_transforms(self):
        res = []
        for tr in self.transforms:
            if tr.always_apply:
                res.append(tr)
        return res

    def __call__(self, force_apply=False, **data):
        need_to_run = force_apply or random.random() < self.p
        transforms = self.transforms if need_to_run else self.get_always_apply_transforms()

        # transformation pipeline
        for tr in transforms:
            data = tr(force_apply, self.targets, **data)

        return data

    def __repr__(self):
        return f'Compose({self.transforms[2:-2]}, {self.p}, {self.targets["img_keywords"]}, ' \
               f'{self.targets["mask_keywords"]}, {self.targets["fmask_keywords"]}, ' \
               f'{self.targets["keypoint_keywords"]}, {self.targets["bbox_keywords"]}, ' \
               f'{self.targets["value_keywords"]}, {self.transforms[-1]})'

