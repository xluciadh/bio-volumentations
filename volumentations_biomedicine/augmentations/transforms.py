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
import numpy as np
from ..core.transforms_interface import DualTransform, ImageOnlyTransform
from ..augmentations import functional as F
from ..random_utils import uniform, sample_range_uniform
from typing import List, Sequence, Tuple, Union
from ..typing import TypeSextetFloat, TypeTripletFloat, TypePairFloat
from .utils import parse_limits, parse_coefs, to_tuple


# Potential upgrade : different sigmas for different channels
class GaussianNoise(ImageOnlyTransform):
    """Adds gaussian noise to the image.

        Noise is drawn from the normal distribution. 

        Args:
            var_limit (tuple, optional): variance of normal distribution is randomly chosen from this interval.
            Defaults to (0.001, 0.1).
            mean (float, optional): mean of normal distribution. Defaults to 0.
            always_apply (bool, optional): always apply transformation in composition. Defaults to False.
            p (float, optional): chance of applying transformation in composition. Defaults to 0.5.

        Targets:
            image
        Image types:
            float32
    """
    def __init__(self, var_limit: tuple = (0.001, 0.1), mean: float = 0,
                 always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.var_limit = var_limit
        self.mean = mean

    def apply(self, img, **params):
        return F.gaussian_noise(img, sigma=params['sigma'], mean=self.mean)

    def get_params(self, **params):
        var = uniform(self.var_limit[0], self.var_limit[1])
        sigma = var ** 0.5
        return {"sigma": sigma}

    def __repr__(self):
        return f'GaussianNoise({self.var_limit}, {self.mean}, {self.always_apply}, {self.p})'


class PoissonNoise(ImageOnlyTransform):
    """Adds poisson noise to the image.
            Args:
            intensity_limit (tuple, optional):
            Defaults to (0.001, 0.1).
    """
    def __init__(self,
                 intensity_limit=(1, 10),
                 always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.intensity_limit = intensity_limit

    def apply(self, img, **params):
        return F.poisson_noise(img, intensity=params['intensity'])

    def get_params(self, **params):
        intensity = uniform(self.intensity_limit[0], self.intensity_limit[1])
        return {"intensity": intensity}

    def __repr__(self):
        return f'PoissonNoise({self.always_apply}, {self.p})'


# TODO anti_aliasing_downsample keep parameter or remove?
class Resize(DualTransform):
    """Resize input to the given shape.

        Resize input using skimage resize function. Shape is expected without channel dimensions. If there is one less
        dimension, than expected then size of last dimension(time) is unchanged. Interpolation, border_mode, ival,
        mval and anti_aliasing_downsample are arguments for
        https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize

        Args:
            shape (tuple of ints): shape of desired image without channel dimension. If inputed with one less
                dimensions, it is expected that it is time dimensions and is copied from image.
            interpolation (int, optional): order of spline interpolation for image. Defaults to 1.
            border_mode (string, optional): points outside image are filled according to this mode.
                Defaults to 'reflect'.
            ival (float, optional): value outside of image when the border_mode is chosen to be "constant".
                Defaults to 0.
            mval (float, optional): value outside of mask when the border_mode is chosen to be "constant".
                Defaults to 0.
            anti_aliasing_downsample (bool, optional): controls if the gaussian filter should be used on image before
                downsampling, recommended. Defaults to True.
            ignore_index (float | None, optional): If ignore_index is float, then transformation of mask is done with 
                border_mode = "constant" and mval = ignore_index. If ignore_index is None, then it does nothing.
                Defaults to None.
            always_apply (bool, optional): always apply transformation in composition. Defaults to False.
            p (float, optional): chance of applying transformation in composition. Defaults to 1.

        Targets:
            image, mask
        Image types:
            float32
    """
    def __init__(self, shape: tuple, interpolation: int = 1, border_mode: str = 'reflect', ival: float = 0,
                 mval: float = 0, anti_aliasing_downsample: bool = True, ignore_index: Union[float, None] = None,
                 always_apply: bool = False, p: float = 1):
        
        super().__init__(always_apply, p)
        self.shape = shape
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.mask_mode = border_mode
        self.ival = ival
        self.mval = mval
        self.anti_aliasing_downsample = anti_aliasing_downsample
        if not (ignore_index is None):
            self.mask_mode = "constant"
            self.mval = ignore_index

    def apply(self, img, **params):
        return F.resize(img, input_new_shape=self.shape, interpolation=self.interpolation,
                        border_mode=self.border_mode, cval=self.ival,
                        anti_aliasing_downsample=self.anti_aliasing_downsample)

    def apply_to_mask(self, mask, **params):
        return F.resize(mask, input_new_shape=self.shape, interpolation=0,
                        border_mode=self.mask_mode, cval=self.mval, anti_aliasing_downsample=False,
                        mask=True)
        
    def __repr__(self):
        return f'Resize({self.shape}, {self.interpolation}, {self.border_mode} , {self.ival}, {self.mval},' \
               f'{self.anti_aliasing_downsample},   {self.always_apply}, {self.p})'


class Scale(DualTransform):
    """Rescale input by the given scale.

        Rescaling is done by function zoom from scipy. If scale_factor is float, spatial dimensions are scaled by this
        number. If it is list, then it is expected without channel dimensions. If there is one less dimension, than
        expected, the size of last dimensions(time) is unchanged.
        Check https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html for additional arguments.

        Args:
            scales (float|List[float], optional): Value by which the input should be scaled. If
                there is single value, then all spatial dimensions are scaled by it. If 
                input is list then all dimensions except for channel one are scaled by it. If 
                there is one less dimensions then last dimension(time) is not scaled. Defaults to 1.
            interpolation (int, optional): order of spline interpolation for image. Defaults to 1.
            border_mode (str, optional): points outside image are filled according to this mode.
                Defaults to 'reflect'.
            ival (float, optional): value outside of image when the border_mode is chosen to be "constant".
                Defaults to 0.
            mval (float, optional): value outside of mask when the border_mode is chosen to be "constant".
                Defaults to 0.
            ignore_index (float | None, optional): If ignore_index is float, then transformation of mask is done with 
                border_mode = "constant" and mval = ignore_index. If ignore_index is None, then it does nothing.
                Defaults to None.
            always_apply (bool, optional): always apply transformation in composition. Defaults to False.
            p (float, optional): chance of applying transformation in composition. Defaults to 1.
        Targets:
            image, mask
        Image types:
            float32
    """
    def __init__(self, scales: Union[float, TypeTripletFloat] = 1,
                 interpolation: str = 'sitkLinear', spacing: Union[float, TypeTripletFloat] = None,
                 border_mode: str = 'reflect', ival: float = 0, mval: float = 0,
                 ignore_index: Union[float, None] = None, always_apply: bool = False, p: float = 1):
        super().__init__(always_apply, p)
        self.scale = parse_coefs(scales, identity_element=1.)
        self.interpolation = interpolation
        self.spacing: TypeTripletFloat = parse_coefs(spacing, identity_element=1.)
        self.border_mode = border_mode
        self.mask_mode = border_mode
        self.ival = ival
        self.mval = mval
        if not (ignore_index is None):
            self.mask_mode = "constant"
            self.mval = ignore_index

    def apply(self, img, **params):
        return F.affine(img,
                        scales=self.scale,
                        interpolation=self.interpolation,
                        border_mode=self.border_mode,
                        value=self.ival,
                        spacing=self.spacing)

    def apply_to_mask(self, mask, **params):
        interpolation = 'sitkNearestNeighbor'
        return F.affine(mask,
                        scales=self.scale,
                        interpolation=interpolation,
                        border_mode=self.mask_mode,
                        value=self.mval,
                        spacing=self.spacing)

    def __repr__(self):
        return f'Scale({self.scale}, {self.interpolation}, {self.border_mode}, {self.ival}, {self.mval},' \
               f'{self.always_apply}, {self.p})'


class RandomScale(DualTransform):
    """Randomly rescale input by the given scale.

        Under the hood, https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html is being used.

        Args:
            scaling_limit (float | Tuple[float] | List[Tuple[float]], optional): Limit of scaling factors.
                If there is single value, then all spatial dimensions are scaled by it. 
                If input is tuple, it creates interval from which the single value for scaling will be chosen. 
                If input is list it should have length of number axes of input - 1 (- channel dimension) and 
                contains tuple of 2 elements. All dimensions except for channel one 
                are scaled by the number from the interval given by tuple. If there is one less dimensions then
                last dimension(time) is not scaled. Defaults to (0.9, 1.1).
            interpolation (int, optional): order of spline interpolation for image. Defaults to 1.
            border_mode (str, optional): points outside image are filled according to the this mode.
                Defaults to 'reflect'.
            ival (float, optional): value outside of image when the border_mode is chosen to be "constant".
                Defaults to 0.
            mval (float, optional): value outside of mask when the border_mode is chosen to be "constant".
                Defaults to 0.
            ignore_index (float | None, optional): If ignore_index is float, then transformation of mask is done with 
                border_mode = "constant" and mval = ignore_index. If ignore_index is None, then it does nothing.
                Defaults to None.
            spacing (TripleFloats | float | None)
            always_apply (bool, optional): always apply transformation in composition. Defaults to False.
            p (float, optional): chance of applying transformation in composition. Defaults to 0.5.
        Targets:
            image, mask
        Image types:
            float32
    """      
    def __init__(self, scaling_limit: Union[float, TypePairFloat, TypeTripletFloat, TypeSextetFloat] = (0.9, 1.1),
                 interpolation: str = 'sitkLinear', spacing: Union[float, TypeTripletFloat] = None,
                 border_mode: str = 'reflect', ival: float = 0, mval: float = 0,
                 ignore_index: Union[float, None] = None, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.scaling_limit: TypeSextetFloat = parse_limits(scaling_limit)
        self.interpolation: str = interpolation
        self.spacing: TypeTripletFloat = parse_coefs(spacing, identity_element=1.)
        self.border_mode = border_mode
        self.mask_mode = border_mode
        self.ival: float = ival
        self.mval: float = mval
        if not (ignore_index is None):
            self.mask_mode = "constant"
            self.mval = ignore_index

    def get_params(self, **data):
        # set parameters of the transform
        scale = sample_range_uniform(self.scaling_limit)

        return {
            "scale": scale,
        }

    def apply(self, img, **params):
        return F.affine(img,
                        scales=params["scale"],
                        interpolation=self.interpolation,
                        border_mode=self.border_mode,
                        value=self.ival,
                        spacing=self.spacing)

    def apply_to_mask(self, mask, **params):

        interpolation = 'sitkNearestNeighbor'
        return F.affine(mask,
                        scales=params["scale"],
                        interpolation=interpolation,
                        border_mode=self.mask_mode,
                        value=self.mval,
                        spacing=self.spacing)

    def __repr__(self):
        return f'RandomScale({self.scaling_limit}, {self.interpolation}, {self.always_apply}, {self.p})'


class RandomRotate90(DualTransform):
    """Rotation of input by 0/90/180/270 degrees in spatial dimensions.

        Input is being rotated around the specified axes. For example if axes = [3,2], 
        then input is rotated around 3 axis (1 and 2 axes are changing)
        and afterward it is rotated around 2 axis(1 and 3 axes are changing).

        Args:
            axes (List[int], optional): list of axes around which input is rotated and also determines 
                order if shuffle_axis is false. Ignoring axes which are not in this list [1,2,3]. 
                Number in axes do not need to be unique. Defaults to [1, 2, 3].
            shuffle_axis (bool, optional): If set to True, order of rotations is random. Defaults to False.
            always_apply (bool, optional): always apply transformation in composition. Defaults to False.
            p (float, optional): chance of applying transformation in composition. Defaults to 0.5.
        Targets:
            image, mask
        Image types:
            float32
    """
    def __init__(self, axes: List[int] = None, shuffle_axis: bool = False,
                 always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.axes = axes
        self.shuffle_axis = shuffle_axis

    def apply(self, img, **params):
        for factor, axes in zip(params["factor"], params["rotation_around"]):
            img = np.rot90(img, factor, axes=axes)
        return img

    def apply_to_mask(self, mask, **params):
        for i in range(len(params["rotation_around"])):
            mask = np.rot90(mask, params["factor"][i], axes=(
                params["rotation_around"][i][0] - 1, params["rotation_around"][i][1] - 1))
        return mask

    def get_params(self, **data):

        # Rotate by all axis by default
        if self.axes is None:
            self.axes = [1, 2, 3]

        # Create all combinations for rotating
        axes_to_rotate = {1: (2, 3), 2: (1, 3), 3: (1, 2)}
        rotation_around = []
        for i in self.axes:
            if i in axes_to_rotate.keys():
                rotation_around.append(axes_to_rotate[i])

        # shuffle order of axis
        if self.shuffle_axis:
            random.shuffle(rotation_around)

        # choose angle to rotate
        factor = [random.randint(0, 3) for _ in range(len(rotation_around))]
        return {"factor": factor,
                "rotation_around": rotation_around}

    def __repr__(self):
        return f'RandomRotate90({self.axes}, {self.always_apply}, {self.p})'


class Flip(DualTransform):
    """Flips input around specified axes. 

        Args:
            axes (List[int], optional): List of axes around which is flip done. Defaults to [1,2,3].
            always_apply (bool, optional): always apply transformation in composition. Defaults to False.
            p (float, optional): chance of applying transformation in composition. Defaults to 1.
        Targets:
            image, mask
        Image types:
            float32
    """
    def __init__(self, axes: List[int] = None, always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.axes = axes

    def apply(self, img, **params):
        return np.flip(img, params["axes"])

    def apply_to_mask(self, mask, **params):
        # Mask has no dimension channel
        return np.flip(mask, axis=[item - 1 for item in params["axes"]])

    def get_params(self, **data):
        if self.axes is None:
            axes = [1, 2, 3]
        else:
            axes = self.axes
        return {"axes": axes}

    def __repr__(self):
        return f'Flip({self.axes}, {self.always_apply}, {self.p})'


class RandomFlip(DualTransform):
    """Flips a input around a tuple of axes randomly chosen from the input list of axis combinations.

        If axes_to_choose to choose is None, random subset of spatial axes is chosen.

        Args:
        
            axes_to_choose (List[Tuple[int]] or None, optional): Randomly chooses tuple of axes from list around 
                which to flip input. If None then a random subset of spatial axes is chosen. Defaults to None.
            always_apply (bool, optional): always apply transformation in composition. Defaults to False.
            p (float, optional): chance of applying transformation in composition. Defaults to 0.5.
        Targets:
            image, mask
        Image types:
            float32
    """
    def __init__(self, axes_to_choose: Union[None, List[Tuple[int]]] = None, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.axes = axes_to_choose

    def apply(self, img, **params):
        return np.flip(img, params["axes"])

    def apply_to_mask(self, mask, **params):
        # Mask has no dimension channel
        return np.flip(mask, axis=[item - 1 for item in params["axes"]])

    def get_params(self, **data):
        
        if self.axes is None or len(self.axes) == 0:
            # Pick random combination of axes to flip
            combinations = [(1,), (2,), (3,), (1, 2),
                            (1, 3), (2, 3), (1, 2, 3)]
            axes = random.choice(combinations)
        else:
            # Pick a random choice from input
            axes = random.choice(self.axes)
        return {"axes": axes}

    def __repr__(self):
        return f'Flip({self.axes}, {self.always_apply}, {self.p})'


class CenterCrop(DualTransform):
    """Crops center region of the input. Size of this crop is given by shape.
          
        Unlike CenterCrop from Albumentations, this transform pads the input in dimensions 
        where the input is smaller than the crop-shape with numpy.pad, for which are border_mode, ival and mval.

        https://numpy.org/doc/stable/reference/generated/numpy.pad.html

        Args:
            shape (Tuple[int]) Final shape of input, expected without first axis of image (representing channels): 
            border_mode (str, optional): border mode used for numpy.pad. Defaults to "reflect".
            ival (Tuple[float], optional): values used for 'constant' or 'linear_ramp' for image. Defaults to (0, 0).
            mval (Tuple[float], optional): values used for 'constant' or 'linear_ramp' for mask. Defaults to (0, 0).
            ignore_index (float | None, optional): If ignore_index is float, then transformation of mask is done with 
                border_mode = "constant" and mval = ignore_index. If ignore_index is None, then it does nothing.
                Defaults to None.
            always_apply (bool, optional): always apply transformation in composition. Defaults to False.
            p (float, optional): chance of applying transformation in composition. Defaults to 1.
        Targets:
            image, mask
        Image types:
            float32
    """
    def __init__(self, shape: Tuple[int], border_mode: str = "reflect", ival: Union[Sequence[float], float] = (0, 0),
                 mval: Union[Sequence[float], float] = (0, 0), ignore_index: Union[float, None] = None,
                 always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply, p)
        self.shape = np.asarray(shape, dtype=np.intc)
        self.border_mode = border_mode
        self.mask_mode = border_mode
        self.ival = ival
        self.mval = mval
        
        if not (ignore_index is None):
            self.mask_mode = "constant"
            self.mval = ignore_index

    def apply(self, img, **params):
        return F.center_crop(img, self.shape, self.border_mode, self.ival, False)

    def apply_to_mask(self, mask, **params):
        return F.center_crop(mask, self.shape, self.mask_mode, self.mval, False)

    def __repr__(self):
        return f'CenterCrop({self.shape}, {self.always_apply}, {self.p})'


class RandomCrop(DualTransform):
    """Randomly crops region from input. Size of this crop is given by shape.

        Unlike RandomCrop from Albumentations, this transform pads the input in dimensions 
        where the input is smaller than the crop-shape with numpy.pad, for which are border_mode, ival and mval.



        Args:
            shape (Tuple[int]) Final shape of input, expected without first axis of image (representing channels): 
            border_mode (str, optional): border mode used for numpy.pad. Defaults to "reflect".
            ival (Tuple[float], optional): values used for 'constant' or 'linear_ramp' for image. Defaults to (0, 0).
            mval (Tuple[float], optional): values used for 'constant' or 'linear_ramp' for mask. Defaults to (0, 0).
            ignore_index (float | None, optional): If ignore_index is float, then transformation of mask is done with 
                border_mode = "constant" and mval = ignore_index. If ignore_index is None, then it does nothing.
                Defaults to None.
            always_apply (bool, optional): always apply transformation in composition. Defaults to False.
            p (float, optional): chance of applying transformation in composition. Defaults to 1.
        Targets:
            image, mask
        Image types:
            float32
    """
    def __init__(self, shape: tuple, border_mode: str = "reflect", ival: Union[Sequence[float], float] = (0, 0),
                 mval: Union[Sequence[float], float] = (0, 0), ignore_index: Union[float, None] = None,
                 always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply, p)
        self.shape = np.asarray(shape, dtype=np.intc)
        self.border_mode = border_mode
        self.mask_mode = border_mode
        self.ival = ival
        self.mval = mval

        if not (ignore_index is None):
            self.mask_mode = "constant"
            self.mval = ignore_index

    def apply(self, img, crop_start=np.array((0, 0, 0))):
        return F.random_crop(img, self.shape, crop_start, self.border_mode, self.ival, mask=False)

    def apply_to_mask(self, mask, crop_start=np.array((0, 0, 0))):
        return F.random_crop(mask, self.shape, crop_start, self.mask_mode, self.mval, mask=True)

    def get_params(self, **data):

        return {
            "crop_start": [random.random() for _ in range(len(self.shape))]
        }

    def __repr__(self):
        return f'RandomCrop({self.shape}, {self.always_apply}, {self.p})'


class RandomAffineTransform(DualTransform):
    """Rotation around spatial axes.

        Rotation around each axis is chosen randomly from given interval in angle_limit. If a float X is given instead,
        for given axis then it becomes interval [-X, X]. If scaling_coef is used, it should be list with length equal 3. 

        Args:
            angle_limit (List[Tuple[float] | float], optional): Contains intervals in degrees from which angle of
                rotation is chosen, for corresponding axis. Defaults to [(-15, 15),(-15, 15),(-15, 15)].
            translation_limit (List[Tuple[int], | int] | None, optional): List of length equal to the number of axes -1
                (minus channel), each element controls translation in this axis. This list consists of intervals,
                from which, it is then randomly chosen the translation vector. Defaults to None.
            scaling_limit (List[Tuple[float] | float): Contains intervals in degrees from which angle of rotation is
                chosen, for corresponding axis. Defaults to [(-15, 15),(-15, 15),(-15, 15)]. Scale 1.2 in axis
            spacing (List[float] | None, optional): List which contains scaling coefficients to make
                the image data isotropic in spatial dimensions. Length of list needs to be 3(number of spatial axes)
                as only spatial dimensions are scaled. If scaling_coef is set to None, there is no scalling.
                Recommended for anisotropic data and if one of spatial axis have significantly lower amount of
                samples. Defaults to None.
            change_to_isotropic (bool, optional): Change data from anisotropic to isotropic. Defaults to False.
            interpolation (Int, optional): The order of spline interpolation. Defaults to 1.
            border_mode (str, optional): The mode parameter determines how the input array is extended beyond its
                boundaries. Defaults to 'constant'.
            ival (float, optional): Value to fill past edges of image if mode is 'constant'. Defaults to 0.
            mval (float, optional): Value to fill past edges of mask if mode is 'constant'. Defaults to 0.
            ignore_index ( float | None, optional): If ignore_index is float, then transformation of mask is done with 
                border_mode = "constant" and mval = ignore_index. If ignore_index is None, then it does nothing.
                Defaults to None.
            always_apply (bool, optional): always apply transformation in composition. Defaults to False.
            p (float, optional): chance of applying transformation in composition. Defaults to 0.5.
        Targets:
            image, mask
        Image types:
            float32
    """
    def __init__(self, angle_limit: Union[float, TypePairFloat, TypeSextetFloat] = (15, 15, 15),
                 translation_limit: Union[float, TypePairFloat, TypeSextetFloat] = (0, 0, 0),
                 scaling_limit: Union[float, TypePairFloat, TypeSextetFloat] = (0.2, 0.2, 0.2),
                 spacing: Union[float, TypeTripletFloat] = None,
                 change_to_isotropic: bool = False,
                 interpolation: str = 'sitkLinear',
                 border_mode: str = 'reflect', ival: float = 0, mval: float = 0, 
                 ignore_index: Union[float, None] = None, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.angle_limit: TypeSextetFloat = parse_limits(angle_limit, identity_element=0)
        self.translation_limit: TypeSextetFloat = parse_limits(translation_limit, identity_element=0)
        self.scaling_limit: TypeSextetFloat = parse_limits(scaling_limit, identity_element=1)
        self.spacing: TypeTripletFloat = parse_coefs(spacing, identity_element=1)
        self.interpolation: str = interpolation        # not used
        self.border_mode = border_mode                 # not used
        self.mask_mode = border_mode                   # not used
        self.ival = ival
        self.mval = mval
        self.keep_scale = not change_to_isotropic

        if ignore_index is not None:
            self.mask_mode = "constant"
            self.mval = ignore_index

    def apply(self, img, **params):
        
        return F.affine(img,
                        scales=params["scale"],
                        degrees=params["angles"],
                        translation=params["translation"],
                        interpolation=self.interpolation,
                        border_mode=self.border_mode,
                        value=self.ival,
                        spacing=self.spacing)

    def apply_to_mask(self, mask, **params):
        
        interpolation = 'sitkNearestNeighbor'
        return F.affine(mask,
                        scales=params["scale"],
                        degrees=params["angles"],
                        translation=params["translation"],
                        interpolation=interpolation,
                        border_mode=self.border_mode,
                        value=self.ival,
                        spacing=self.spacing)

    def get_params(self, **data):

        # set parameters of the transform
        scales = sample_range_uniform(self.scaling_limit)
        angles = sample_range_uniform(self.angle_limit)
        translation = sample_range_uniform(self.translation_limit)

        return {
            "scale": scales,
            "angles": angles,
            "translation": translation
        }


class AffineTransform(DualTransform):
    """Rotation around spatial axes.

        Rotation around each axis is chosen randomly from given interval in angle_limit. If a float X is given instead
        for given axis then it becomes interval [-X, X]. If scaling_coef is used, it should be list with length equal 3.

        Args:
            angles (List[Tuple[float] | float], optional): Contains intervals in degrees from which angle of
                rotation is chosen, for corresponding axis. Defaults to (0, 0, 0).
            translation (List[Tuple[int], | int] | None, optional): List of length equal to the number of axes -1
                (minus channel), each element controls translation in this axis. This list consists of intervals,
                from which, it is then randomly chosen the translation vector. Defaults to (0, 0, 0).
            scale (List[Tuple[float]] | float): Contains intervals in degrees from which angle of rotation is
                chosen, for corresponding axis. Defaults to (1, 1, 1).
            spacing (List[float] | None, optional): List which contains scaling coefficients to make
                the image data isotropic in spatial dimensions. Length of list needs to be 3(number of spatial axes)
                as only spatial dimensions are scaled. If scaling_coef is set to None, there is no scalling.
                Recommended for anisotropic data and if one of spatial axis have significantly lower amount of
                samples. Defaults to (1, 1, 1).
            change_to_isotropic (bool, optional): Change data from anisotropic to isotropic. Defaults to False.
            interpolation (Int, optional): The order of spline interpolation. Defaults to 1.
            border_mode (str, optional): The mode parameter determines how the input array is extended beyond its
                boundaries. Defaults to 'constant'.
            ival (float, optional): Value to fill past edges of image if mode is 'constant'. Defaults to 0.
            mval (float, optional): Value to fill past edges of mask if mode is 'constant'. Defaults to 0.
            ignore_index ( float | None, optional): If ignore_index is float, then transformation of mask is done with
                border_mode = "constant" and mval = ignore_index. If ignore_index is None, then it does nothing.
                Defaults to None.
            always_apply (bool, optional): always apply transformation in composition. Defaults to False.
            p (float, optional): chance of applying transformation in composition. Defaults to 0.5.
        Targets:
            image, mask
        Image types:
            float32
    """
    def __init__(self, angles: TypeTripletFloat = (0, 0, 0),
                 translation: TypeTripletFloat = (0, 0, 0),
                 scale: TypeTripletFloat = (1, 1, 1),
                 spacing: TypeTripletFloat = (1, 1, 1),
                 change_to_isotropic: bool = False,
                 interpolation: str = 'sitkLinear',
                 border_mode: str = 'reflect', ival: float = 0, mval: float = 0,
                 ignore_index: Union[float, None] = None, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.angles: TypeTripletFloat = parse_coefs(angles, identity_element=0)
        self.translation: TypeTripletFloat = parse_coefs(translation, identity_element=0)
        self.scale: TypeTripletFloat = parse_coefs(scale, identity_element=1)
        self.spacing: TypeTripletFloat = parse_coefs(spacing, identity_element=1)
        self.interpolation: str = interpolation        # not used
        self.border_mode = border_mode                 # not used
        self.mask_mode = border_mode                   # not used
        self.ival = ival
        self.mval = mval
        self.keep_scale = not change_to_isotropic

        if ignore_index is not None:
            self.mask_mode = "constant"
            self.mval = ignore_index

    def apply(self, img, **params):
        return F.affine(img,
                        scales=self.scale,
                        degrees=self.angles,
                        translation=self.translation,
                        interpolation=self.interpolation,
                        border_mode=self.border_mode,
                        value=self.ival,
                        spacing=self.spacing)

    def apply_to_mask(self, mask, **params):

        interpolation = 'sitkNearestNeighbor'
        return F.affine(mask,
                        scales=self.scale,
                        degrees=self.angles,
                        translation=self.translation,
                        interpolation=interpolation,
                        border_mode=self.border_mode,
                        value=self.ival,
                        spacing=self.spacing)


# TODO create checks (mean, std, got good shape, and etc.), what if given list but only one channel, and reverse.
class NormalizeMeanStd(ImageOnlyTransform):
    """Normalization of image by given mean and std.

        For a single channel image normalization is applied by the formula :math:`img = (img - mean) / std`.
        
        If image contains more channels, then for each channel previous formula is used.

        Args:
            mean (float | List[float]): Mean of image. If there are more channels, then it should be list of means
                for each channel.
            std (float | List[float]): Std of image. If there are more channels, then it should be list of stds
                for each channel.
            always_apply (bool, optional): always apply transformation in composition. Defaults to False.
            p (float, optional): chance of applying transformation in composition. Defaults to 1.
        Targets:
            image
        Image types:
            float32
    """
    def __init__(self, mean: Union[List[float], float], std: Union[List[float], float],
                 always_apply: bool = True, p: float = 1.0):
        super().__init__(always_apply, p)
        self.mean = np.array(mean, dtype=np.float32) 
        self.std = np.array(std, dtype=np.float32) 
        self.denominator = np.reciprocal(self.std, dtype=np.float32)

    def apply(self, image, **params):
        return F.normalize_mean_std(image, self.mean, self.denominator)

    def __repr__(self):
        return f'NormalizeMeanStd({self.mean}, {self.std}, ' \
               f' {self.always_apply}, {self.p})'


class GaussianBlur(ImageOnlyTransform):
    """Performs gaussian blur on the image.

        Sigma parameter determines the strength of gaussian blur. There is no blurring between channels. 
        By default, there is no blurring also on time dimension. If given single number, channels and axes are blurred
        with same strength. If given tuple, blurring is performed with same effect over channels, but on each axis
        differently. If given List, each channel is blurred differently, according to the element inside list.

        For more information about border_mode and cval check scipy.ndimage.gaussian_filter.

        Args:
            sigma (float, Tuple(float), List[Tuple(float) | float] , optional): Determines strength of the blurring. 
                List must have length equal to the number of channels. Tuple should have same number elements as
                number of axes - 1. Defaults to 0.8.
            border_mode (str, optional): The mode parameter determines how the input array is extended beyond its
                boundaries. Defaults to "reflect".
            cval (float, optional):  Value to fill past edges of image if mode is 'constant'. Defaults to 0.
            always_apply (bool, optional): always apply transformation in composition. Defaults to False.
            p (float, optional): chance of applying transformation in composition. Defaults to 0.5.
        Targets:
            image
        Image types:
            float32
    """
    def __init__(self, sigma: Union[float , Tuple[float], List[ Union[Tuple[float], float]]] = 0.8,
                 border_mode: str = "reflect", cval: float = 0,
                 always_apply: bool = False, p: float = 0.5):
        
        super().__init__(always_apply, p)
        self.sigma = sigma
        self.border_mode = border_mode
        self.cval = cval

    def apply(self, img, **params):
        return F.gaussian_blur(img, self.sigma, self.border_mode, self.cval)


class RandomGaussianBlur(ImageOnlyTransform):
    """Performs gaussian blur on the image with a random strength blurring.

        Behaves similarly to GaussianBlur, sigma has same format, but each number in sigma creates 
        interval [start_of_interval, sigma_number], from which random number is chosen. 

        Args:
            max_sigma (float, Tuple(float), List[Tuple(float) | float, optional): Determines end of interval from which
                strength of blurring is chosen. Defaults to 0.8.
            start_of_interval (float, optional): Determines start of interval from which strength of blurring is chosen.
                Defaults to 0.
            border_mode (str, optional): The mode parameter determines how the input array is extended beyond its
                boundaries. Defaults to "reflect".
            cval (float, optional):  Value to fill past edges of image if mode is 'constant'. Defaults to 0.
            always_apply (bool, optional): always apply transformation in composition. Defaults to False.
            p (float, optional): chance of applying transformation in composition. Defaults to 0.5.
        Targets:
            image
        Image types:
            float32
    """
    def __init__(self, max_sigma: Union[float, TypeTripletFloat] = 0.8,
                 start_of_interval: float = 0, border_mode: str = "reflect", cval: float = 0,
                 always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.max_sigma = parse_coefs(max_sigma)
        self.start_of_interval = start_of_interval
        self.border_mode = border_mode
        self.cval = cval

    def apply(self, img, **params):
        return F.gaussian_blur(img, params["sigma"], self.border_mode, self.cval)

    def get_params(self, **data):
        if isinstance(self.sigma, (float, int)):
            sigma = random.uniform(self.start_of_interval, self.sigma)
        elif isinstance(self.sigma, tuple):
            sigma = tuple([random.uniform(self.start_of_interval, self.sigma[i]) for i in range(len(self.sigma))])
        else:
            sigma = []
            for channel in self.sigma:
                if isinstance(channel, (float, int)):
                    sigma.append(random.uniform(self.start_of_interval, channel))
                else:
                    sigma.append(tuple([random.uniform(self.start_of_interval, channel) for i in range(len(channel))]))
        return {"sigma": sigma}


class RandomGamma(ImageOnlyTransform):
    """Performs gamma transform with a randomly selected gamma.

        Gamma is randomly selected from interval given by gamma_limit. If the values in image are not in [0,1] interval
        then this transformation is skipped.


        Args:
            gamma_limit (Tuple(float), optional): Interval from which gamma is selected. Defaults to (0.8, 1.20).
            always_apply (bool, optional): always apply transformation in composition. Defaults to False.
            p (float, optional): chance of applying transformation in composition. Defaults to 0.5.
        Targets:
            image
        Image types:
            float32
    """
    def __init__(self, gamma_limit: Tuple[float] = (0.8, 1.20),
                 always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.gamma_limit = gamma_limit

    def apply(self, img, gamma=1, **params):
        return F.gamma_transform(img, gamma=gamma)

    def get_params(self, **data):
        return {"gamma": random.uniform(self.gamma_limit[0], self.gamma_limit[1])}

    @staticmethod
    def get_transform_init_args_names():
        return "gamma_limit", "eps"

    def __repr__(self):
        return f'RandomGamma({self.gamma_limit}, {self.always_apply}, {self.p})'


class RandomBrightnessContrast(ImageOnlyTransform):
    """Randomly change brightness and contrast of the input image.

        Unlike RandomBrightnessContrast from Albumentations, this transform is using
        formula  :math:`f(a) = (c+1) * a + b`, where c is contrast and b is brightness.

        Args:
            brightness_limit ((float, float) | float, optional): Interval from which change in brightness is taken.
                If limit is a single float, the interval will be (-limit, limit). If change in brightness is 0,
                brightness won`t change. Defaults to 0.2.
            contrast_limit ((float, float) | float, optional): Interval from which change in contrast is taken.
                If limit is a single float, the interval will be (-limit, limit). If change in contrast is 0,
                contrast won`t change. Defaults to 0.2.
            always_apply (bool, optional): always apply transformation in composition. Defaults to False.
            p (float, optional): chance of applying transformation in composition. Defaults to 0.5.
        Targets:
            image
        Image types:
            float32
    """
    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, always_apply=False, p=0.5,):
        super().__init__(always_apply, p)
        self.brightness_limit = to_tuple(brightness_limit)
        self.contrast_limit = to_tuple(contrast_limit)

    def apply(self, img, **params):
        return F.brightness_contrast_adjust(img, params['alpha'], params['beta'])

    def get_params(self, **data):
        return {
            "alpha": 1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1]),
            "beta": 0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1]),
        }

    @staticmethod
    def get_transform_init_args_names():
        return "brightness_limit", "contrast_limit"

    def __repr__(self):
        return f'RandomBrightnessContrast({self.brightness_limit}, {self.contrast_limit},  ' \
               f'{self.always_apply}, {self.p})'


class HistogramEqualization(ImageOnlyTransform):
    """Performs equalization of histogram.

        This equalization is done channel-wise, meaning that each channel is equalized separately. 
        Images are normalized over both spatial and temporal domains together. The output is in the range [0,1].

        This transformation is performed with
        https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_hist

        Args:
            bins (int, optional): Number of bins for image histogram. Defaults to 256.
            always_apply (bool, optional): always apply transformation in composition. Defaults to False.
            p (float, optional): chance of applying transformation in composition. Defaults to 1.
        Targets:
            image
        Image types:
            float32
    """
    def __init__(self, bins: int = 256, always_apply: bool = False, p: float = 1):
        super().__init__(always_apply, p)
        self.bins = bins

    def apply(self, img, **params):
        return F.histogram_equalization(img, self.bins)


class Pad(DualTransform):
    """Pads the input.

        Input is padded based on pad_size. If pad_size is only one number, all spatial axes are padded on both sides
        with this number. If it is tuple, then it has same behaviour as pad_size except sides are padded with different 
        number of pixels. If it is List, then it must have 3 items, which define padding for each spatial dimension
        separately (in either of the ways described above). If the List is shorter, remaining axes are padded with 0. 

        For other parameters check https://numpy.org/doc/stable/reference/generated/numpy.pad.html    

        Args:
            pad_size (int | Tuple[int] | List[int | Tuple[int]]): Determines number of pixels to be padded.
                Tuple should be of size 2. List should be of size equal to the image axes - 1 (channel axis).
            border_mode (str, optional): numpy.pad parameter . Defaults to 'constant'.
            ival (float | Sequence, optional): value for image if needed by chosen border_mode. Defaults to 0.
            mval (float | Sequence, optional): value for mask if needed by chosen border_mode. Defaults to 0.
            ignore_index ( float | None, optional): If ignore_index is float, then transformation of mask is done with 
                border_mode = "constant" and mval = ignore_index. If ignore_index is None, then it does nothing.
                Defaults to None.
            always_apply (bool, optional): always apply transformation in composition. Defaults to False.
            p (float, optional): chance of applying transformation in composition. Defaults to 0.5.
        Targets:
            image, mask
        Image types:
            float32
    """
    def __init__(self, pad_size: Union[int, Tuple[int],  List[Union[int, Tuple[int]]]], border_mode: str = 'constant',
                 ival: Union[float, Sequence] = 0, mval: Union[float, Sequence] = 0,
                 ignore_index: Union[float, None] = None, always_apply: bool = False, p : float = 1):
        super().__init__(always_apply, p)
        self.pad_size = pad_size
        self.border_mode = border_mode
        self.mask_mode = border_mode 
        self.ival = ival
        self.mval = mval

        if not (ignore_index is None):
            self.mask_mode = "constant"
            self.mval = ignore_index

    def apply(self, img, **parms):
        return F.pad_pixels(img, self.pad_size, self.border_mode, self.ival)

    def apply_to_mask(self, mask, **params):
        return F.pad_pixels(mask, self.pad_size, self.mask_mode, self.mval, True)

    def __repr__(self):
        return f'Pad({self.pad_size}, {self.border_mode}, {self.ival}, {self.mval}, {self.always_apply}, ' \
               f'{self.p})'


class Normalize(ImageOnlyTransform):
    """Normalize image channels to the given mean and std.

        Normalization is performed channel-wise. 

        Args:
            mean (float | List[float], optional): Value of desired mean. If it is list, then it should have
             same length as number of channels, and each value corresponds to the desired mean in respective channel. Defaults to 0.
            std (float | List[float], optional): Value of desired std. If it is list, then it should have
             same length as number of channels, and each value corresponds to the desired std in respective channel. Defaults to 1.
            always_apply (bool, optional): always apply transformation in composition. Defaults to False.
            p (float, optional): chance of applying transformation in composition. Defaults to 1.
        Targets:
            image
        Image types:
            float32
    """
    def __init__(self, mean: Union[float, List[float]] = 0, std: Union[float, List[float]] = 1,
                 always_apply: bool = True, p: float = 1.0):
        super().__init__(always_apply, p)
        self.mean = mean
        self.std = std

    def apply(self, img, **params):
        return F.normalize(img, self.mean, self.std)

    def __repr__(self):
        return f'Normalize({self.mean}, {self.std}, {self.always_apply}, {self.p})'


class Contiguous(DualTransform):
    def apply(self, image):
        return np.ascontiguousarray(image)

    def __repr__(self):
        return f'Contiguous()'



####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
# taken transforms from forks, not implemented #####################
# shouldnt work, except for Float() ################################
####################################################################
####################################################################
####################################################################



# not sure what should be end goal for this transformation 
# but currecntly used as a transform which is always called at the start of Compose
class Float(DualTransform):
    def apply(self, image):
        # TODO this should change value range to (0,1) from the original dtype' value range. Have a look at the code
        #  below and function F.to_float(), F.from_float()

        # if isinstance(image, np.float32):
        #     # assume the image already has correct value range (0,1)
        #     return image
        #
        # # change value range to (0,1)
        # if issubclass(image.dtype.type, numbers.Integral):
        #     dtype_info = np.iinfo(image.dtype)
        # else:
        #     dtype_info = np.finfo(image.dtype)
        #
        # if dtype_info.min == 0:
        #     return image.astype(np.float32) / dtype_info.max
        #
        # return ((image.astype(np.float32) / (-dtype_info.min)) / 2) + 0.5

        return image.astype(np.float32)

    def __repr__(self):
        return f'Float()'


class ElasticTransform(DualTransform):
    def __init__(self, deformation_limits=(0, 0.25), interpolation=1, border_mode='constant', value=0, mask_value=0,
                 always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.deformation_limits = deformation_limits
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, sigmas, alphas, random_state=None):
        return F.elastic_transform(img, sigmas, alphas, interpolation=self.interpolation, random_state=random_state,
                                   border_mode=self.border_mode, value=self.value)

    def apply_to_mask(self, img, sigmas, alphas, random_state=None):
        return F.elastic_transform(img, sigmas, alphas, interpolation=0, random_state=random_state,
                                   border_mode=self.border_mode, value=self.mask_value)

    def get_params(self, **data):
        image = data["image"]  # [H, W, D]
        random_state = random.randint(0, 10000)
        deformation = random.uniform(*self.deformation_limits)
        sigmas = [deformation * x for x in image.shape[:3]]
        alphas = [random.uniform(x / 8, x / 2) for x in sigmas]
        return {
            "random_state": random_state,
            "sigmas": sigmas,
            "alphas": alphas,
        }

    def __repr__(self):
        return f'ElasticTransform({self.deformation_limits}, {self.interpolation}, {self.border_mode}, {self.value}, ' \
               f'{self.mask_value}, {self.always_apply}, {self.p})'


# TODO from shape variables to shape arrays

#Its here, so it wont show in documentation.

    """GridDropout, drops out rectangular regions of an image and the corresponding mask in a grid fashion.
        Args:
        ratio (float): the ratio of the mask holes to the unit_size (same for horizontal and vertical directions).
            Must be between 0 and 1. Default: 0.5.
        unit_size_min (int): minimum size of the grid unit. Must be between 2 and the image shorter edge.
            If 'None', holes_number_x and holes_number_y are used to setup the grid. Default: `None`.
        unit_size_max (int): maximum size of the grid unit. Must be between 2 and the image shorter edge.
            If 'None', holes_number_x and holes_number_y are used to setup the grid. Default: `None`.
        holes_number_x (int): the number of grid units in x direction. Must be between 1 and image width//2.
            If 'None', grid unit width is set as image_width//10. Default: `None`.
        holes_number_y (int): the number of grid units in y direction. Must be between 1 and image height//2.
            If `None`, grid unit height is set equal to the grid unit width or image height, whatever is smaller.
        holes_number_z (int): the number of grid units in z direction. Must be between 1 and image depth//2.
            If `None`, grid unit depth is set equal to the grid unit width or image height, whatever is smaller.
        shift_x (int): offsets of the grid start in x direction from (0,0) coordinate.
            Clipped between 0 and grid unit_width - hole_width. Default: 0.
        shift_y (int): offsets of the grid start in y direction from (0,0) coordinate.
            Clipped between 0 and grid unit height - hole_height. Default: 0.
        shift_z (int): offsets of the grid start in z direction from (0,0) coordinate.
            Clipped between 0 and grid unit depth - hole_depth. Default: 0.
        random_offset (boolean): weather to offset the grid randomly between 0 and grid unit size - hole size
            If 'True', entered shift_x, shift_y, shift_z are ignored and set randomly. Default: `False`.
        fill_value (int): value for the dropped pixels. Default = 0
        mask_fill_value (int): value for the dropped pixels in mask.
            If `None`, tranformation is not applied to the mask. Default: `None`.
    Targets:
        image, mask
    Image types:
        uint8, float32
    References:
        https://arxiv.org/abs/2001.04086
    """

class GridDropout(DualTransform):

    def __init__(
            self,
            ratio: float = 0.5,
            unit_size_min: int = None,
            unit_size_max: int = None,
            holes_number_x: int = None,
            holes_number_y: int = None,
            holes_number_z: int = None,
            shift_x: int = 0,
            shift_y: int = 0,
            shift_z: int = 0,
            random_offset: bool = False,
            fill_value: int = 0,
            mask_fill_value: int = None,
            always_apply: bool = False,
            p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.ratio = ratio
        self.unit_size_min = unit_size_min
        self.unit_size_max = unit_size_max
        self.holes_number_x = holes_number_x
        self.holes_number_y = holes_number_y
        self.holes_number_z = holes_number_z
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.shift_z = shift_z
        self.random_offset = random_offset
        self.fill_value = fill_value
        self.mask_fill_value = mask_fill_value
        if not 0 < self.ratio <= 1:
            raise ValueError("ratio must be between 0 and 1.")

    def apply(self, image, holes=(), **params):
        return F.cutout(image, holes, self.fill_value)

    def apply_to_mask(self, image, holes=(), **params):
        if self.mask_fill_value is None:
            return image

        return F.cutout(image, holes, self.mask_fill_value)

    def get_params(self, **data):
        img = data["image"]
        height, width, depth = img.shape[:3]
        # set grid using unit size limits
        if self.unit_size_min and self.unit_size_max:
            if not 2 <= self.unit_size_min <= self.unit_size_max:
                raise ValueError(
                    "Max unit size should be >= min size, both at least 2 pixels.")
            if self.unit_size_max > min(height, width):
                raise ValueError(
                    "Grid size limits must be within the shortest image edge.")
            unit_width = random.randint(
                self.unit_size_min, self.unit_size_max + 1)
            unit_height = unit_width
            unit_depth = unit_width
        else:
            # set grid using holes numbers
            if self.holes_number_x is None:
                unit_width = max(2, width // 10)
            else:
                if not 1 <= self.holes_number_x <= width // 2:
                    raise ValueError(f"The hole_number_x must be between 1 and image width//2 ({width//2}), "
                                     f"but was {self.holes_number_x}.")
                unit_width = width // self.holes_number_x
            if self.holes_number_y is None:
                unit_height = max(min(unit_width, height), 2)
            else:
                if not 1 <= self.holes_number_y <= height // 2:
                    raise ValueError(f"The hole_number_y must be between 1 and image height//2 ({height//2}), "
                                     f"but was {self.holes_number_y}.")
                unit_height = height // self.holes_number_y
            if self.holes_number_z is None:
                unit_depth = max(min(unit_height, depth), 2)
            else:
                if not 1 <= self.holes_number_z <= depth // 2:
                    raise ValueError(f"The hole_number_z must be between 1 and image depth//2 ({depth//2}), "
                                     f"but was {self.holes_number_z}.")
                unit_depth = depth // self.holes_number_z

        hole_width = int(unit_width * self.ratio)
        hole_height = int(unit_height * self.ratio)
        hole_depth = int(unit_depth * self.ratio)
        # min 1 pixel and max unit length - 1
        hole_width = min(max(hole_width, 1), unit_width - 1)
        hole_height = min(max(hole_height, 1), unit_height - 1)
        hole_depth = min(max(hole_depth, 1), unit_depth - 1)
        # set offset of the grid
        if self.shift_x is None:
            shift_x = 0
        else:
            shift_x = min(max(0, self.shift_x), unit_width - hole_width)
        if self.shift_y is None:
            shift_y = 0
        else:
            shift_y = min(max(0, self.shift_y), unit_height - hole_height)
        if self.shift_z is None:
            shift_z = 0
        else:
            shift_z = min(max(0, self.shift_z), unit_depth - hole_depth)
        if self.random_offset:
            shift_x = random.randint(0, unit_width - hole_width)
            shift_y = random.randint(0, unit_height - hole_height)
            shift_z = random.randint(0, unit_depth - hole_depth)
        holes = []
        for i in range(width // unit_width + 1):
            for j in range(height // unit_height + 1):
                for k in range(depth // unit_depth + 1):
                    x1 = min(shift_x + unit_width * i, width)
                    y1 = min(shift_y + unit_height * j, height)
                    z1 = min(shift_z + unit_depth * j, depth)
                    x2 = min(x1 + hole_width, width)
                    y2 = min(y1 + hole_height, height)
                    z2 = min(z1 + hole_depth, depth)
                    holes.append((x1, y1, z1, x2, y2, z2))

        return {"holes": holes}

    def get_transform_init_args_names(self):
        return (
            "ratio",
            "unit_size_min",
            "unit_size_max",
            "holes_number_x",
            "holes_number_y",
            "shift_x",
            "shift_y",
            "mask_fill_value",
            "random_offset",
        )

    def __repr__(self):
        return f'GridDropout({self.ratio}, {self.unit_size_min}, {self.unit_size_max}, {self.holes_number_x}, ' \
               f'{self.holes_number_y}, {self.holes_number_z}, {self.shift_x}, {self.shift_y}, {self.shift_z}, ' \
               f'{self.random_offset}, {self.fill_value}, {self.mask_fill_value}, {self.always_apply}, {self.p})'


class RandomDropPlane(DualTransform):
    """Randomly drop some planes in axis randomly chosen from 'axes' input array.

    Args:
        plane_drop_prob (float): float value in (0.0, 1.0) range. Default: 0.1
        axes (tuple). Default: 0
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask

    Image types:
        uint8, float32
    """

    def __init__(
            self,
            plane_drop_prob=0.1,
            axes=(0,),
            always_apply=False,
            p=1.0
    ):
        super().__init__(always_apply, p)
        self.plane_drop_prob = plane_drop_prob
        self.axes = axes

    def get_params(self, **data):
        img = data["image"]
        # TODO add choosing axis randomly from all spatial axes like in flip if axes is initially None
        axis = random.choice(self.axes)
        r = img.shape[axis]
        indexes = []
        for i in range(r):
            if random.uniform(0, 1) > self.plane_drop_prob:
                indexes.append(i)
        if len(indexes) == 0:
            indexes.append(0)

        return {
            "indexes": indexes, "axis": axis,
        }

    def apply(self, img, indexes=(), axis=0, **params):
        return np.take(img, indexes, axis=axis)

    def apply_to_mask(self, mask, indexes=(), axis=0, **params):
        return np.take(mask, indexes, axis=axis)

    def __repr__(self):
        return f'RandomDropPlane({self.plane_drop_prob}, {self.axes}, {self.always_apply}, {self.p})'

