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

import numpy as np
from ..biovol_typing import TypeTripletFloat
from typing import Sequence, Optional
import SimpleITK as sitk

from .utils import get_image_center, ras_to_lps, np_to_sitk, sitk_to_np

DEBUG = False

SITK_interpolation = {
    0: 'sitkNearestNeighbor',
    1: 'sitkLinear'
}


def get_affine_transform(domain_limit,
                         scales: TypeTripletFloat,
                         degrees: TypeTripletFloat,
                         translation: TypeTripletFloat,
                         spacing: TypeTripletFloat,
                         keep_scale: bool = True) -> sitk.Euler3DTransform:
    # copy arrays
    scaling = np.asarray(scales)
    rotation = np.asarray(degrees)
    translation = np.asarray(translation)  # * np.asarray(spacing)

    center_lps = get_image_center(domain_limit,
                                  spacing=spacing,
                                  lps=False)

    scaling_transform = get_scaling_transform(
        scaling,
        center_lps=center_lps,
    )

    if DEBUG:
        print('domain_limit', domain_limit)
        print('center_lps', center_lps)
        print('translation', translation)
        print('scaling', scaling)
        print('spacing', spacing)

    rotation_transform = get_rotation_transform(
        rotation,
        translation,
        center_lps=center_lps,
    )

    transforms = [scaling_transform,
                  rotation_transform]

    transform = sitk.CompositeTransform(transforms)
    transform = transform.GetInverse()

    return transform


def get_scaling_transform(
        scaling_params: Sequence[float],
        center_lps: Optional[TypeTripletFloat] = None,
) -> sitk.ScaleTransform:

    # 1.5 means the objects look 1.5 times larger
    transform = sitk.ScaleTransform(3)
    scaling_params_array = np.array(scaling_params).astype(float)
    transform.SetScale(scaling_params_array)

    # set center
    if center_lps is not None:
        transform.SetCenter(center_lps)
    return transform


def get_rotation_transform(
        degrees: Sequence[float],
        translation: Sequence[float],
        center_lps: Optional[TypeTripletFloat] = None,
) -> sitk.Euler3DTransform:

    transform = sitk.Euler3DTransform()
    radians = np.radians(degrees).tolist()

    # SimpleITK uses LPS
    radians_lps = ras_to_lps(radians)
    translation_lps = ras_to_lps(translation)

    if DEBUG:
        print('radians_lps', radians_lps)
        print('translation_lps', translation_lps)
        print('center_lps', center_lps)

    transform.SetRotation(*radians_lps)
    transform.SetTranslation(translation_lps)

    # set center
    if center_lps is not None:
        transform.SetCenter(center_lps)

    return transform.GetInverse()


def apply_sitk_transform(
        image: np.array,
        sitk_transform: sitk.Euler3DTransform,
        interpolation: str,
        default_value: float,
        spacing: TypeTripletFloat = (1., 1., 1.)
) -> np.array:

    assert len(image.shape) >= 4, f'image.shape: {image.shape}'

    # resolve the image shape
    ch = image.shape[0]
    fr = 1 if len(image.shape) == 4 else image.shape[4]

    if len(image.shape) == 4:
        image_expanded = np.expand_dims(image, 4)
        expanded = True
    else:
        image_expanded = image
        expanded = False

    # convert numpy array to sitk image
    sitk_image = np_to_sitk(image_expanded)
    sitk_image.SetSpacing(spacing)

    # apply transform
    floating = reference = sitk_image
    interpolator = sitk.ResampleImageFilter()
    interpolator.SetInterpolator(getattr(sitk, interpolation))
    interpolator.SetReferenceImage(reference)
    interpolator.SetDefaultPixelValue(float(default_value))
    interpolator.SetTransform(sitk_transform)

    resampled = interpolator.Execute(floating)
    np_array = sitk_to_np(resampled, channels=ch, frames=fr)
    if expanded:
        np_array = np_array.squeeze(4)

    assert image.shape == np_array.shape, f"image.shape: {image.shape} np_array.shape:, {np_array.shape}"

    # np_array = np_array.transpose()  # ITK to NumPy convention
    return np_array
