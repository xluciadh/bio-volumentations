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
from ..augmentations.spatial_funcional import get_affine_transform
from ..random_utils import uniform, sample_range_uniform
from typing import List, Sequence, Tuple, Union
from ..biovol_typing import TypeSextetFloat, TypeTripletFloat, TypePairFloat, TypeSpatioTemporalCoordinate,\
    TypeSextetInt, TypeSpatialCoordinate, TypeSpatialShape
from .utils import parse_limits, parse_coefs, parse_pads, to_tuple, validate_bbox, get_spatio_temporal_domain_limit,\
    to_spatio_temporal


# TODO anti_aliasing_downsample keep parameter or remove?
class Resize(DualTransform):
    """Resize input to the given shape.

        Internally, the ``skimage.transform.resize`` function is used.
        The ``interpolation``, ``border_mode``, ``ival``, ``mval``,
        and ``anti_aliasing_downsample`` arguments are forwarded to it. More details at:
        https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize.

        Args:
            shape (tuple of ints): The desired image shape.

                Must be of either of: ``(Z, Y, X)`` or ``(Z, Y, X, T)``.

                The unspecified dimensions (C and possibly T) are not affected.
            interpolation (int, optional): Order of spline interpolation.

                Defaults to ``1``.
            border_mode (str, optional): Values outside image domain are filled according to this mode.

                Defaults to ``'reflect'``.
            ival (float, optional): Value of `image` voxels outside of the `image` domain. Only applied when ``border_mode = 'constant'``.

                Defaults to ``0``.
            mval (float, optional): Value of `mask` voxels outside of the `mask` domain. Only applied when ``border_mode = 'constant'``.
            
                Defaults to ``0``.
            anti_aliasing_downsample (bool, optional): Controls if the Gaussian filter should be applied before
                downsampling. Recommended. 
                
                Defaults to ``True``.
            ignore_index (float | None, optional): If a float, then transformation of `mask` is done with 
                ``border_mode = 'constant'`` and ``mval = ignore_index``. 
                
                If ``None``, this argument is ignored.
                
                Defaults to ``None``.
            always_apply (bool, optional): Always apply this transformation in composition. 
            
                Defaults to ``False``.
            p (float, optional): Chance of applying this transformation in composition. 
            
                Defaults to ``1``.

        Targets:
            image, mask, float_mask
    """
    def __init__(self, shape: tuple, interpolation: int = 1, border_mode: str = 'reflect', ival: float = 0,
                 mval: float = 0, anti_aliasing_downsample: bool = True, ignore_index: Union[float, None] = None,
                 always_apply: bool = False, p: float = 1):
        
        super().__init__(always_apply, p)
        self.shape: TypeSpatioTemporalCoordinate = to_spatio_temporal(shape)
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

    def apply_to_float_mask(self, mask, **params):
        return F.resize(mask, input_new_shape=self.shape, interpolation=self.interpolation,
                        border_mode=self.mask_mode, cval=self.mval, anti_aliasing_downsample=False,
                        mask=True)

    def apply_to_keypoints(self, keypoints, **params):
        return F.resize_keypoints(keypoints,
                                  domain_limit=params['domain_limit'],
                                  new_shape=self.shape)

    """
    def apply_to_bboxes(self, bboxes, **params):
        for bbox in bboxes:
            new_bbox = F.resize_keypoints(bbox,
                                          input_new_shape=self.shape,
                                          original_shape=params['original_shape'],
                                          keep_all=True)

            if validate_bbox(bbox, new_bbox, min_overlay_ratio):
                res.append(new_bbox)

        return res
    """

    def get_params(self, **data):

        # read shape of the original image
        domain_limit: TypeSpatioTemporalCoordinate = get_spatio_temporal_domain_limit(data)

        return {
            "domain_limit": domain_limit,
        }
        
    def __repr__(self):
        return f'Resize({self.shape}, {self.interpolation}, {self.border_mode} , {self.ival}, {self.mval},' \
               f'{self.anti_aliasing_downsample},   {self.always_apply}, {self.p})'


class Scale(DualTransform):
    """Rescale input by the given scale.

        Args:
            scales (float|List[float], optional): Value by which the input should be scaled.

                Must be either of: ``S``, ``[S_Z, S_Y, S_X]``, or ``[S_Z, S_Y, S_X, S_T]``.

                If a float, then all spatial dimensions are scaled by it (equivalent to ``[S, S, S]``).

                The unspecified dimensions (C and possibly T) are not affected.

                Defaults to ``1``.
            interpolation (int, optional): Order of spline interpolation.

                Defaults to ``1``.
            spacing (float | Tuple[float, float, float] | None, optional): Voxel spacing for individual spatial dimensions.

                Must be either of: ``S``, ``(S1, S2, S3)``, or ``None``.

                If ``None``, equivalent to ``(1, 1, 1)``.

                If a float ``S``, equivalent to ``(S, S, S)``.

                Otherwise, a scale for each spatial dimension must be given.

                Defaults to ``None``.
            border_mode (str, optional): Values outside image domain are filled according to this mode.

                Defaults to ``'constant'``.
            ival (float, optional): Value of `image` voxels outside of the `image` domain. Only applied when ``border_mode = 'constant'``.

                Defaults to ``0``.
            mval (float, optional): Value of `mask` voxels outside of the `mask` domain. Only applied when ``border_mode = 'constant'``.

                Defaults to ``0``.
            ignore_index (float | None, optional): If a float, then transformation of `mask` is done with 
                ``border_mode = 'constant'`` and ``mval = ignore_index``. 
                
                If ``None``, this argument is ignored.

                Defaults to ``None``.
            always_apply (bool, optional): Always apply this transformation in composition. 
            
                Defaults to ``False``.
            p (float, optional): Chance of applying this transformation in composition. 
            
                Defaults to ``1``.

        Targets:
            image, mask, float_mask
    """
    def __init__(self, scales: Union[float, TypeTripletFloat] = 1,
                 interpolation: int = 1, spacing: Union[float, TypeTripletFloat] = None,
                 border_mode: str = 'constant', ival: float = 0, mval: float = 0,
                 ignore_index: Union[float, None] = None, always_apply: bool = False, p: float = 1):
        super().__init__(always_apply, p)
        self.scale = parse_coefs(scales, identity_element=1.)
        self.interpolation: int = interpolation
        self.spacing: TypeTripletFloat = parse_coefs(spacing, identity_element=1.)
        self.border_mode = border_mode              # not implemented
        self.mask_mode = border_mode                # not implemented
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
        interpolation = 0   # refers to 'sitkNearestNeighbor'
        return F.affine(np.expand_dims(mask, 0),
                        scales=self.scale,
                        interpolation=interpolation,
                        border_mode=self.mask_mode,
                        value=self.mval,
                        spacing=self.spacing)[0]

    def apply_to_float_mask(self, mask, **params):
        return F.affine(np.expand_dims(mask, 0),
                        scales=self.scale,
                        interpolation=self.interpolation,
                        border_mode=self.mask_mode,
                        value=self.mval,
                        spacing=self.spacing)[0]

    def apply_to_keypoints(self, keypoints, **params):
        return F.affine_keypoints(keypoints,
                                  scales=self.scale,
                                  spacing = self.spacing,
                                  domain_limit=params['domain_limit'])

    """
    def apply_to_bboxes(self, bboxes, **params):
        for bbox in bboxes:
            new_bbox = F.affine_keypoints(bbox,
                                          scales=self.scale,
                                          domain_limit=params['domain_limit'],
                                          spacing = self.spacing,
                                          keep_all=True)

            if validate_bbox(bbox, new_bbox):
                res.append(new_bbox)

        return res
    """

    def get_params(self, **data):
        domain_limit: TypeSpatioTemporalCoordinate = get_spatio_temporal_domain_limit(data)
        return {'domain_limit': domain_limit}

    def __repr__(self):
        return f'Scale({self.scale}, {self.interpolation}, {self.border_mode}, {self.ival}, {self.mval},' \
               f'{self.always_apply}, {self.p})'


# TODO cannot rescale T dimension
class RandomScale(DualTransform):
    """Randomly rescale input.

        Args:
            scaling_limit (float | Tuple[float] | List[Tuple[float]], optional): Limits of scaling factors.

                Must be either of: ``S``, ``(S1, S2)``, ``(S_Z, S_Y, S_X)``, or ``(S_Z1, S_Z2, S_Y1, S_Y2, S_X1, S_X2)``.

                If a float ``S``, then all spatial dimensions are scaled by a random number drawn uniformly from
                the interval [1-S, 1+S] (equivalent to inputting ``(1-S, 1+S, 1-S, 1+S, 1-S, 1+S)``).

                If a tuple of 2 floats, then all spatial dimensions are scaled by a random number drawn uniformly
                from the interval [S1, S2] (equivalent to inputting ``(S1, S2, S1, S2, S1, S2)``).

                If a tuple of 3 floats, then an interval [1-S_a, 1+S_a] is constructed for each spatial
                dimension and the scale is randomly drawn from it
                (equivalent to inputting ``(1-S_Z, 1+S_Z, 1-S_Y, 1+S_Y, 1-S_X, 1+S_X)``).

                If a tuple of 6 floats, the scales for individual spatial dimensions are randomly drawn from the
                respective intervals [S_Z1, S_Z2], [S_Y1, S_Y2], [S_X1, S_X2].

                The unspecified dimensions (C and T) are not affected.

                Defaults to ``(0.9, 1.1)``.

            interpolation (int, optional): Order of spline interpolation.

                Defaults to ``1``.

            spacing (float | Tuple[float, float, float] | None, optional): Voxel spacing for individual spatial dimensions.

                Must be either of: ``S``, ``(S1, S2, S3)``, or ``None``.

                If ``None``, equivalent to ``(1, 1, 1)``.

                If a float ``S``, equivalent to ``(S, S, S)``.

                Otherwise, a scale for each spatial dimension must be given.

                Defaults to ``None``.

            border_mode (str, optional): Values outside image domain are filled according to the mode.

                Defaults to ``'constant'``.

            ival (float, optional): Value of `image` voxels outside of the `image` domain. Only applied when ``border_mode = 'constant'``.

                Defaults to ``0``.

            mval (float, optional): Value of `mask` voxels outside of the `mask` domain. Only applied when ``border_mode = 'constant'``.

                Defaults to ``0``.

            ignore_index (float | None, optional): If a float, then transformation of `mask` is done with 
                ``border_mode = 'constant'`` and ``mval = ignore_index``. 
                
                If ``None``, this argument is ignored.

                Defaults to ``None``.

            always_apply (bool, optional): Always apply this transformation in composition.

                Defaults to ``False``.

            p (float, optional): Chance of applying this transformation in composition.

                Defaults to ``0.5``.

        Targets:
            image, mask, float_mask
    """      
    def __init__(self, scaling_limit: Union[float, TypePairFloat, TypeTripletFloat, TypeSextetFloat] = (0.9, 1.1),
                 interpolation: int = 1, spacing: Union[float, TypeTripletFloat] = None,
                 border_mode: str = 'constant', ival: float = 0, mval: float = 0,
                 ignore_index: Union[float, None] = None, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.scaling_limit: TypeSextetFloat = parse_limits(scaling_limit)
        self.interpolation: int = interpolation
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
        domain_limit: TypeSpatioTemporalCoordinate = get_spatio_temporal_domain_limit(data)
        scale = sample_range_uniform(self.scaling_limit)

        return {
            "domain_limit": domain_limit,
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
        interpolation = 0   # refers to 'sitkNearestNeighbor'
        return F.affine(np.expand_dims(mask, 0),
                        scales=params["scale"],
                        interpolation=interpolation,
                        border_mode=self.mask_mode,
                        value=self.mval,
                        spacing=self.spacing)[0]

    def apply_to_float_mask(self, mask, **params):
        return F.affine(np.expand_dims(mask, 0),
                        scales=params["scale"],
                        interpolation=self.interpolation,
                        border_mode=self.mask_mode,
                        value=self.mval,
                        spacing=self.spacing)[0]

    def apply_to_keypoints(self, keypoints, **params):
        return F.affine_keypoints(keypoints,
                                  scales=params["scale"],
                                  spacing=self.spacing,
                                  domain_limit=params['domain_limit'])

    def __repr__(self):
        return f'RandomScale({self.scaling_limit}, {self.interpolation}, {self.always_apply}, {self.p})'


class RandomRotate90(DualTransform):
    """Rotation of input by 0, 90, 180, or 270 degrees around the specified spatial axes.

        Args:
            axes (List[int], optional): List of axes around which the input is rotated. Recognised axis symbols are
                ``1`` for Z, ``2`` for Y, and ``3`` for X. A single axis can occur multiple times in the list.
                If ``shuffle_axis = False``, the order of axes determines the order of transformations.

                Defaults to ``[1, 2, 3]``.
            shuffle_axis (bool, optional): If set to ``True``, the order of rotations is random.

                Defaults to ``False``.
            always_apply (bool, optional): Always apply this transformation in composition. 
            
                Defaults to ``False``.
            p (float, optional): Chance of applying this transformation in composition. 
            
                Defaults to ``0.5``.

        Targets:
            image, mask, float_mask
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
        for rot, factor in zip(params["rotation_around"], params["factor"]):
            mask = np.rot90(mask, factor, axes=(rot[0] - 1, rot[1] - 1))
        return mask

    def apply_to_keypoints(self, keypoints, keep_all=True, **params):
        for rot, factor in zip(params["rotation_around"], params["factor"]):
            keypoints = F.rot90_keypoints(keypoints,
                                          factor=factor,
                                          axes=(rot[0] - 1, rot[1] - 1),
                                          img_shape=params['img_shape'])
        return keypoints

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
        img_shape = np.array(data['image'].shape[1:4])

        return {"factor": factor,
                "rotation_around": rotation_around,
                "img_shape": img_shape}

    def __repr__(self):
        return f'RandomRotate90({self.axes}, {self.always_apply}, {self.p})'


class Flip(DualTransform):
    """Flip input around the specified spatial axes.

        Args:
            axes (List[int], optional): List of axes around which is flip done. Recognised axis symbols are
                ``1`` for Z, ``2`` for Y, and ``3`` for X.

                Defaults to ``[1,2,3]``.
            always_apply (bool, optional): Always apply this transformation in composition. 
            
                Defaults to ``False``.
            p (float, optional): Chance of applying this transformation in composition. 
            
                Defaults to ``1``.

        Targets:
            image, mask, float_mask
    """
    def __init__(self, axes: List[int] = None, always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.axes = axes

    def apply(self, img, **params):
        return np.flip(img, params["axes"])

    def apply_to_mask(self, mask, **params):
        # Mask has no dimension channel
        return np.flip(mask, axis=[item - 1 for item in params["axes"]])

    def apply_to_keypoints(self, keypoints, keep_all=False, **params):
        return F.flip_keypoints(keypoints,
                                axes=params['axes'],
                                img_shape=params['img_shape'])

    def get_params(self, **data):
        axes = [1, 2, 3] if self.axes is None else self.axes
        img_shape = np.array(data['image'].shape[1:4])
        return {"axes": axes,
                "img_shape": img_shape}

    def __repr__(self):
        return f'Flip({self.axes}, {self.always_apply}, {self.p})'


# TODO include possibility to pick empty combination = no flipping
class RandomFlip(DualTransform):
    """Flip input around a set of axes randomly chosen from the input list of axis combinations.

        Args:
            axes_to_choose (List[Tuple[int]] or None, optional): List of axis indices from which one option
                is randomly chosen. Recognised axis symbols are ``1`` for Z, ``2`` for Y, and ``3`` for X.
                The image will be flipped around all axes in the chosen combination.

                If ``None``, a random subset of spatial axes is chosen, corresponding to inputting
                ``[(,), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]``.

                Defaults to ``None``.
            always_apply (bool, optional): Always apply this transformation in composition. 
            
                Defaults to ``False``.
            p (float, optional): Chance of applying this transformation in composition. 
            
                Defaults to ``0.5``.

        Targets:
            image, mask, float_mask
    """
    def __init__(self, axes_to_choose: Union[None, List[Tuple[int]]] = None, always_apply=False, p=0.5):
        super().__init__(always_apply, p)

        # TODO: check if input value `axes_to_choice` valid
        self.axes = axes_to_choose

    def apply(self, img, **params):
        return np.flip(img, params["axes"])

    def apply_to_mask(self, mask, **params):
        # Mask has no dimension channel
        return np.flip(mask, axis=[item - 1 for item in params["axes"]])

    def apply_to_keypoints(self, keypoints, keep_all=False, **params):
        return F.flip_keypoints(keypoints,
                                axes=params['axes'],
                                img_shape=params['img_shape'])

    def get_params(self, **data):
        
        to_choose = [1, 2, 3] if self.axes is None else self.axes
        axes = random.sample(to_choose, random.randint(0, len(to_choose)))
        img_shape = np.array(data['image'].shape[1:4])
        return {"axes": axes,
                "img_shape": img_shape}

    def __repr__(self):
        return f'Flip({self.axes}, {self.always_apply}, {self.p})'


class CenterCrop(DualTransform):
    """Crops the central region of the input of given size.
          
        Unlike ``CenterCrop`` from `Albumentations`, this transform pads the input in dimensions
        where the input is smaller than the ``shape`` with ``numpy.pad``. The ``border_mode``, ``ival`` and ``mval``
        arguments are forwarded to ``numpy.pad`` if padding is necessary. More details at:
        https://numpy.org/doc/stable/reference/generated/numpy.pad.html.

        Args:
            shape (Tuple[int]): The desired shape of input.

                Must be either of: ``[Z, Y, X]`` or ``[Z, Y, X, T]``.
            border_mode (str, optional): Values outside image domain are filled according to this mode.

                Defaults to ``'reflect'``.
            ival (float | Sequence, optional): Values of `image` voxels outside of the `image` domain.
                Only applied when ``border_mode = 'constant'`` or ``border_mode = 'linear_ramp'``.

                Defaults to ``(0, 0)``.
            mval (float | Sequence, optional): Values of `mask` voxels outside of the `mask` domain.
                Only applied when ``border_mode = 'constant'`` or ``border_mode = 'linear_ramp'``.

                Defaults to ``(0, 0)``.
            ignore_index (float | None, optional): If a float, then transformation of `mask` is done with 
                ``border_mode = 'constant'`` and ``mval = ignore_index``. 
                
                If ``None``, this argument is ignored.

                Defaults to ``None``.
            always_apply (bool, optional): Always apply this transformation in composition. 
            
                Defaults to ``False``.
            p (float, optional): Chance of applying this transformation in composition. 
            
                Defaults to ``1``.

        Targets:
            image, mask, float_mask
    """
    def __init__(self, shape: Tuple[int], border_mode: str = "reflect", ival: Union[Sequence[float], float] = (0, 0),
                 mval: Union[Sequence[float], float] = (0, 0), ignore_index: Union[float, None] = None,
                 always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply, p)
        self.output_shape = np.asarray(shape, dtype=np.intc)  # TODO: make it len 3
        self.border_mode = border_mode
        self.mask_mode = border_mode
        self.ival = ival
        self.mval = mval
        
        if not (ignore_index is None):
            self.mask_mode = "constant"
            self.mval = ignore_index

    def apply(self, img, **params):
        return F.crop(img,
                      crop_shape=self.output_shape,
                      crop_position=params['crop_position'],
                      pad_dims=params['pad_dims'],
                      border_mode=self.mask_mode, cval=self.mval, mask=False)

    def apply_to_mask(self, mask, **params):
        return F.crop(mask,
                      crop_shape=self.output_shape,
                      crop_position=params['crop_position'],
                      pad_dims=params['pad_dims'],
                      border_mode=self.mask_mode, cval=self.mval, mask=True)

    def apply_to_keypoints(self, keypoints, keep_all=False, **params):
        return F.crop_keypoints(keypoints,
                                crop_shape=self.output_shape,
                                crop_position=params['crop_position'],
                                pad_dims=params['pad_dims'],
                                keep_all=keep_all)

    def get_params(self, **data):
        # get crop coordinates, position of the corner closest to the image origin
        img_spatial_shape = np.array(data['image'].shape[1:4])
        position: TypeSpatialCoordinate = (img_spatial_shape - self.output_shape) // 2
        position = np.maximum(position, 0).astype(int)
        pad_dims = F.get_pad_dims(img_spatial_shape, self.output_shape)

        return {'crop_position': position,
                'pad_dims': pad_dims}

    def __repr__(self):
        return f'CenterCrop({self.output_shape}, {self.always_apply}, {self.p})'


class RandomCrop(DualTransform):
    """Randomly crops a region of given size from the input.

        Unlike ``RandomCrop`` from `Albumentations`, this transform pads the input in dimensions
        where the input is smaller than the ``shape`` with ``numpy.pad``. The ``border_mode``, ``ival`` and ``mval``
        arguments are forwarded to ``numpy.pad`` if padding is necessary. More details at:
        https://numpy.org/doc/stable/reference/generated/numpy.pad.html.

        Args:
            shape (Tuple[int]): The desired shape of input.

                Must be either of: ``[Z, Y, X]`` or ``[Z, Y, X, T]``.
            border_mode (str, optional): Values outside image domain are filled according to this mode.

                Defaults to ``'reflect'``.
            ival (float | Sequence, optional): Values of `image` voxels outside of the `image` domain.
                Only applied when ``border_mode = 'constant'`` or ``border_mode = 'linear_ramp'``.

                Defaults to ``(0, 0)``.
            mval (float | Sequence, optional): Values of `mask` voxels outside of the `mask` domain.
                Only applied when ``border_mode = 'constant'`` or ``border_mode = 'linear_ramp'``.

                Defaults to ``(0, 0)``.
            ignore_index (float | None, optional): If a float, then transformation of `mask` is done with 
                ``border_mode = 'constant'`` and ``mval = ignore_index``. 
                
                If ``None``, this argument is ignored.

                Defaults to ``None``.
            always_apply (bool, optional): Always apply this transformation in composition. 
            
                Defaults to ``False``.
            p (float, optional): Chance of applying this transformation in composition. 
            
                Defaults to ``1``.

        Targets:
            image, mask, float_mask
    """
    def __init__(self, shape: tuple, border_mode: str = "reflect", ival: Union[Sequence[float], float] = (0, 0),
                 mval: Union[Sequence[float], float] = (0, 0), ignore_index: Union[float, None] = None,
                 always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply, p)
        self.output_shape = np.asarray(shape, dtype=np.intc)
        self.border_mode = border_mode
        self.mask_mode = border_mode
        self.ival = ival
        self.mval = mval

        if not (ignore_index is None):
            self.mask_mode = "constant"
            self.mval = ignore_index

    def apply(self, img, **params):
        return F.crop(img,
                      crop_shape=self.output_shape,
                      crop_position=params['crop_position'],
                      pad_dims=params['pad_dims'],
                      border_mode=self.mask_mode, cval=self.mval, mask=False)

    def apply_to_mask(self, mask, **params):
        return F.crop(mask,
                      crop_shape=self.output_shape,
                      crop_position=params['crop_position'],
                      pad_dims=params['pad_dims'],
                      border_mode=self.mask_mode, cval=self.mval, mask=True)

    def apply_to_keypoints(self, keypoints, keep_all=False, **params):
        return F.crop_keypoints(keypoints,
                                crop_shape=self.output_shape,
                                crop_position=params['crop_position'],
                                pad_dims=params['pad_dims'],
                                keep_all=keep_all)

    def get_params(self, **data):
        # get crop coordinates, position of the corner closest to the image origin
        img_spatial_shape = np.array(data['image'].shape[1:4])
        ranges: TypeSpatialShape = np.maximum(img_spatial_shape - self.output_shape, 0)
        position = np.array([random.randint(0, r) for r in ranges])
        pad_dims = F.get_pad_dims(img_spatial_shape, self.output_shape)
        return {'crop_position': position,
                'pad_dims': pad_dims}

    def __repr__(self):
        return f'RandomCrop({self.output_shape}, {self.always_apply}, {self.p})'


class RandomAffineTransform(DualTransform):
    """Affine transformation of the input image with randomly chosen parameters.

        Args:
            angle_limit (Tuple[float] | float, optional): Intervals in degrees from which angles of
                rotation for the spatial axes are chosen.

                Must be either of: ``A``, ``(A1, A2)``, or ``(A_Z1, A_Z2, A_Y1, A_Y2, A_X1, A_X2)``.

                If a float, equivalent to ``(-A, A, -A, A, -A, A)``.

                If a tuple with 2 items, equivalent to ``(A1, A2, A1, A2, A1, A2)``.

                If a tuple with 6 items, angle of rotation is randomly chosen from an interval [A_a1, A_a2] for each
                spatial axis.

                Defaults to ``(15, 15, 15)``.
            translation_limit (Tuple[int] | int | None, optional): Intervals from which the translation parameters
                for the spatial axes are chosen.

                Must be either of: ``T``, ``(T1, T2)``, or ``(T_Z1, T_Z2, T_Y1, T_Y2, T_X1, T_X2)``.

                If a float, equivalent to ``(-T, T, -T, T, -T, T)``.

                If a tuple with 2 items, equivalent to ``(T1, T2, T1, T2, T1, T2)``.

                If a tuple with 6 items, the translation parameter is randomly chosen from an interval [T_a1, T_a2] for
                each spatial axis.

                Defaults to ``(0, 0, 0)``.
            scaling_limit (Tuple[float] | float, optional): Intervals from which the scales for the spatial axes are chosen.

                Must be either of: ``S``, ``(S1, S2)``, or ``(S_Z1, S_Z2, S_Y1, S_Y2, S_X1, S_X2)``.

                If a float, equivalent to ``(1-S, 1+S, 1-S, 1+S, 1-S, 1+S)``.

                If a tuple with 2 items, equivalent to ``(S1, S2, S1, S2, S1, S2)``.

                If a tuple with 6 items, the scale is randomly chosen from an interval [S_a1, S_a2] for
                each spatial axis.

                Defaults to ``(0.2, 0.2, 0.2)``.
            spacing (float | Tuple[float, float, float] | None, optional): Voxel spacing for individual spatial dimensions.

                Must be either of: ``S``, ``(S1, S2, S3)``, or ``None``.

                If ``None``, equivalent to ``(1, 1, 1)``.

                If a float ``S``, equivalent to ``(S, S, S)``.

                Otherwise, a scale for each spatial dimension must be given.

                Defaults to ``None``.
            change_to_isotropic (bool, optional): Change data from anisotropic to isotropic.

                Defaults to ``False``.
            interpolation (int, optional): Order of spline interpolation.

                Defaults to ``1``.
            border_mode (str, optional): Values outside image domain are filled according to this mode.

                Defaults to ``'constant'``.
            ival (float, optional): Value of `image` voxels outside of the `image` domain. Only applied when ``border_mode = 'constant'``.

                Defaults to ``0``.
            mval (float, optional): Value of `mask` voxels outside of the `mask` domain. Only applied when ``border_mode = 'constant'``.

                Defaults to ``0``.
            ignore_index (float | None, optional): If a float, then transformation of `mask` is done with 
                ``border_mode = 'constant'`` and ``mval = ignore_index``. 
                
                If ``None``, this argument is ignored.

                Defaults to ``None``.
            always_apply (bool, optional): Always apply this transformation in composition. 
            
                Defaults to ``False``.
            p (float, optional): Chance of applying this transformation in composition. 
            
                Defaults to ``0.5``.

        Targets:
            image, mask, float_mask
    """
    def __init__(self, angle_limit: Union[float, TypePairFloat, TypeSextetFloat] = (15., 15., 15.),
                 translation_limit: Union[float, TypePairFloat, TypeSextetFloat] = (0., 0., 0.),
                 scaling_limit: Union[float, TypePairFloat, TypeSextetFloat] = (0.2, 0.2, 0.2),
                 spacing: Union[float, TypeTripletFloat] = None,
                 change_to_isotropic: bool = False,
                 interpolation: int = 1,
                 border_mode: str = 'constant', ival: float = 0, mval: float = 0,
                 ignore_index: Union[float, None] = None, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.angle_limit: TypeSextetFloat = parse_limits(angle_limit, identity_element=0)
        self.translation_limit: TypeSextetFloat = parse_limits(translation_limit, identity_element=0)
        self.scaling_limit: TypeSextetFloat = parse_limits(scaling_limit, identity_element=1)
        self.spacing: TypeTripletFloat = parse_coefs(spacing, identity_element=1)
        self.interpolation: int = interpolation
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
        interpolation = 0   # refers to 'sitkNearestNeighbor'
        return F.affine(np.expand_dims(mask, 0),
                        scales=params["scale"],
                        degrees=params["angles"],
                        translation=params["translation"],
                        interpolation=interpolation,
                        border_mode=self.mask_mode,
                        value=self.mval,
                        spacing=self.spacing)[0]

    def apply_to_float_mask(self, mask, **params):
        return F.affine(np.expand_dims(mask, 0),
                        scales=params["scale"],
                        degrees=params["angles"],
                        translation=params["translation"],
                        interpolation=self.interpolation,
                        border_mode=self.mask_mode,
                        value=self.mval,
                        spacing=self.spacing)[0]

    def apply_to_keypoints(self, keypoints, **params):
        return F.affine_keypoints(keypoints,
                                  scales=params["scale"],
                                  degrees=params["angles"],
                                  translation=params["translation"],
                                  spacing=self.spacing,
                                  domain_limit=params['domain_limit'])

    def get_params(self, **data):

        # set parameters of the transform
        scales = sample_range_uniform(self.scaling_limit)
        angles = sample_range_uniform(self.angle_limit)
        translation = sample_range_uniform(self.translation_limit)
        domain_limit = get_spatio_temporal_domain_limit(data)

        return {
            "scale": scales,
            "angles": angles,
            "translation": translation,
            "domain_limit": domain_limit
        }


class AffineTransform(DualTransform):
    """Affine transformation of the input image with given parameters.

        Args:
            angles (Tuple[float], optional): Angles of rotation for the spatial axes.

                Must be: ``(A_Z, A_Y, A_X)``.

                Defaults to ``(0, 0, 0)``.
            translation (Tuple[float], optional): Translation vector for the spatial axes.

                Must be: ``(T_Z, T_Y, T_X)``.

                Defaults to ``(0, 0, 0)``.
            scale (Tuple[float], optional): Scales for the spatial axes.

                Must be: ``(S_Z, S_Y, S_X)``.

                Defaults to ``(1, 1, 1)``.
            spacing (Tuple[float, float, float], optional): Voxel spacing for individual spatial dimensions.

                Must be: ``(S1, S2, S3)`` (a scale for each spatial dimension must be given).

                Defaults to ``(1, 1, 1)``.
            change_to_isotropic (bool, optional): Change data from anisotropic to isotropic.

                Defaults to ``False``.
            interpolation (int, optional): Order of spline interpolation.

                Defaults to ``1``.
            border_mode (str, optional): Values outside image domain are filled according to this mode.

                Defaults to ``'constant'``.
            ival (float, optional): Value of `image` voxels outside of the `image` domain. Only applied when ``border_mode = 'constant'``.

                Defaults to ``0``.
            mval (float, optional): Value of `mask` voxels outside of the `mask` domain. Only applied when ``border_mode = 'constant'``.

                Defaults to ``0``.
            ignore_index (float | None, optional): If a float, then transformation of `mask` is done with 
                ``border_mode = 'constant'`` and ``mval = ignore_index``. 
                
                If ``None``, this argument is ignored.

                Defaults to ``None``.
            always_apply (bool, optional): Always apply this transformation in composition. 
            
                Defaults to ``False``.
            p (float, optional): Chance of applying this transformation in composition. 
            
                Defaults to ``0.5``.

        Targets:
            image, mask, float_mask
    """
    def __init__(self, angles: TypeTripletFloat = (0, 0, 0),
                 translation: TypeTripletFloat = (0, 0, 0),
                 scale: TypeTripletFloat = (1, 1, 1),
                 spacing: TypeTripletFloat = (1, 1, 1),
                 change_to_isotropic: bool = False,
                 interpolation: int = 1,
                 border_mode: str = 'constant', ival: float = 0, mval: float = 0,
                 ignore_index: Union[float, None] = None, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.angles: TypeTripletFloat = parse_coefs(angles, identity_element=0)
        self.translation: TypeTripletFloat = parse_coefs(translation, identity_element=0)
        self.scale: TypeTripletFloat = parse_coefs(scale, identity_element=1)
        self.spacing: TypeTripletFloat = parse_coefs(spacing, identity_element=1)
        self.interpolation: int = interpolation
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
        interpolation = 0   # refers to 'sitkNearestNeighbor'
        return F.affine(np.expand_dims(mask, 0),
                        scales=self.scale,
                        degrees=self.angles,
                        translation=self.translation,
                        interpolation=interpolation,
                        border_mode=self.mask_mode,
                        value=self.mval,
                        spacing=self.spacing)[0]

    def apply_to_float_mask(self, mask, **params):
        return F.affine(np.expand_dims(mask, 0),
                        scales=self.scale,
                        degrees=self.angles,
                        translation=self.translation,
                        interpolation=self.interpolation,
                        border_mode=self.mask_mode,
                        value=self.mval,
                        spacing=self.spacing)[0]

    def apply_to_keypoints(self, keypoints, **params):
        return F.affine_keypoints(keypoints,
                                  scales=self.scale,
                                  degrees=self.angles,
                                  translation=self.translation,
                                  spacing=self.spacing,
                                  domain_limit=params['domain_limit'])

    def get_params(self, **data):

        # set parameters of the transform
        domain_limit = get_spatio_temporal_domain_limit(data)

        return {
            "domain_limit": domain_limit
        }


# IMAGE ONLY TRANSFORMS
# TODO potential upgrade : different sigmas for different channels
class GaussianNoise(ImageOnlyTransform):
    """Adds Gaussian noise to the image. The noise is drawn from normal distribution with given parameters.

        Args:
            var_limit (tuple, optional): Variance of normal distribution is randomly chosen from this interval.

                Defaults to ``(0.001, 0.1)``.
            mean (float, optional): Mean of normal distribution.

                Defaults to ``0``.
            always_apply (bool, optional): Always apply this transformation in composition.

                Defaults to ``False``.
            p (float, optional): Chance of applying this transformation in composition.

                Defaults to ``0.5``.

        Targets:
            image
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
    """Adds Poisson noise to the image.

        Args:
            intensity_limit (tuple): Range to sample the expected intensity of Poisson noise.

                Defaults to ``(1, 10)``.
            always_apply (bool, optional): Always apply this transformation in composition.

                Defaults to ``False``.
            p (float, optional): Chance of applying this transformation in composition.

                Defaults to ``0.5``.

        Targets:
            image
    """

    def __init__(self,
                 peak_limit=(0.1, 0.5),
                 always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.peak_limit = peak_limit

    def apply(self, img, **params):
        return F.poisson_noise(img, peak=params['peak'])

    def get_params(self, **params):
        peak = uniform(self.peak_limit[0], self.peak_limit[1])
        return {"peak": peak}

    def __repr__(self):
        return f'PoissonNoise({self.always_apply}, {self.p})'


# TODO create checks (mean, std, got good shape, and etc.), what if given list but only one channel, and reverse.
class NormalizeMeanStd(ImageOnlyTransform):
    """Normalize image values to have mean 0 and standard deviation 1, given channel-wise means and standard deviations.

        For a single-channel image, the normalization is applied by the formula: :math:`img = (img - mean) / std`.
        If the image contains more channels, then the previous formula is used for each channel separately.

        It is recommended to input dataset-wide means and standard deviations.

        Args:
            mean (float | List[float]): Channel-wise image mean.

                Must be either of: ``M``, ``(M_1, M_2, ..., M_C)``.
            std (float | List[float]): Channel-wise image standard deviation.

                Must be either of: ``S``, ``(S_1, S_2, ..., S_C)``.
            always_apply (bool, optional): Always apply this transformation in composition. 
            
                Defaults to ``True``.
            p (float, optional): Chance of applying this transformation in composition. 
            
                Defaults to ``1``.

        Targets:
            image
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
    """Performs Gaussian blurring of the image. In case of a multi-channel image, individual channels are blured separately.

        Internally, the ``scipy.ndimage.gaussian_filter`` function is used. The ``border_mode`` and ``cval``
        arguments are forwarded to it. More details at:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html.

        Args:
            sigma (float, Tuple(float), List[Tuple(float) | float] , optional): Gaussian sigma.

                Must be either of: ``S``, ``(S_Z, S_Y, S_X)``, ``(S_Z, S_Y, S_X, S_T)``, ``[S_1, S_2, ..., S_C]``,
                ``[(S_Z1, S_Y1, S_X1), (S_Z2, S_Y2, S_X2), ..., (S_ZC, S_YC, S_XC)]``, or
                ``[(S_Z1, S_Y1, S_X1, S_T1), (S_Z2, S_Y2, S_X2, S_T2), ..., (S_ZC, S_YC, S_XC, S_TC)]``.

                If a float, the spatial dimensions are blurred with the same strength (equivalent to ``(S, S, S)``).

                If a tuple, the sigmas for spatial dimensions and possibly the time dimension must be specified.

                If a list, sigmas for each channel must be specified either as a single number or as a tuple.

                Defaults to ``0.8``.
            border_mode (str, optional): Values outside image domain are filled according to this mode.

                Defaults to ``'reflect'``.
            cval (float, optional): Value to fill past edges of image. Only applied when ``border_mode = 'constant'``.

                Defaults to ``0``.
            always_apply (bool, optional): Always apply this transformation in composition. 
            
                Defaults to ``False``.
            p (float, optional): Chance of applying this transformation in composition. 
            
                Defaults to ``0.5``.

        Targets:
            image
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
    """Performs Gaussian blur on the image with a random strength blurring.
        In case of a multi-channel image, individual channels are blured separately.

        Behaves similarly to GaussianBlur. The Gaussian sigma is randomly drawn from
        the interval [min_sigma, s] for the respective s from ``max_sigma`` for each channel and dimension.

        Internally, the ``scipy.ndimage.gaussian_filter`` function is used. The ``border_mode`` and ``cval``
        arguments are forwarded to it. More details at:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html.

        Args:
            max_sigma (float, Tuple(float), List[Tuple(float) | float] , optional): Maximum Gaussian sigma.

                Must be either of: ``S``, ``(S_Z, S_Y, S_X)``, ``(S_Z, S_Y, S_X, S_T)``, ``[S_1, S_2, ..., S_C]``,
                ``[(S_Z1, S_Y1, S_X1), (S_Z2, S_Y2, S_X2), ..., (S_ZC, S_YC, S_XC)]``, or
                ``[(S_Z1, S_Y1, S_X1, S_T1), (S_Z2, S_Y2, S_X2, S_T2), ..., (S_ZC, S_YC, S_XC, S_TC)]``.

                If a float, the spatial dimensions are blurred equivalently (equivalent to ``(S, S, S)``).

                If a tuple, the sigmas for spatial dimensions and possibly the time dimension must be specified.

                If a list, sigmas for each channel must be specified either as a single number or as a tuple.

                Defaults to ``0.8``.
            min_sigma (float, optional): Minimum Gaussian sigma for all channels and dimensions.

                Defaults to ``0``.
            border_mode (str, optional): Values outside image domain are filled according to this mode.

                Defaults to ``'reflect'``.
            cval (float, optional): Value to fill past edges of image. Only applied when ``border_mode = 'constant'``.

                Defaults to ``0``.
            always_apply (bool, optional): Always apply this transformation in composition. 
            
                Defaults to ``False``.
            p (float, optional): Chance of applying this transformation in composition. 
            
                Defaults to ``0.5``.

        Targets:
            image
    """
    def __init__(self, max_sigma: Union[float, TypeTripletFloat] = 0.8,
                 min_sigma: float = 0, border_mode: str = "reflect", cval: float = 0,
                 always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.max_sigma = parse_coefs(max_sigma)
        self.min_sigma = min_sigma
        self.border_mode = border_mode
        self.cval = cval

    def apply(self, img, **params):
        return F.gaussian_blur(img, params["sigma"], self.border_mode, self.cval)

    def get_params(self, **data):
        if isinstance(self.max_sigma, (float, int)):
            sigma = random.uniform(self.min_sigma, self.max_sigma)
        elif isinstance(self.max_sigma, tuple):
            sigma = tuple([random.uniform(self.min_sigma, self.max_sigma[i]) for i in range(len(self.max_sigma))])
        else:
            sigma = []
            for channel in self.max_sigma:
                if isinstance(channel, (float, int)):
                    sigma.append(random.uniform(self.min_sigma, channel))
                else:
                    sigma.append(tuple([random.uniform(self.min_sigma, channel) for i in range(len(channel))]))
        return {"sigma": sigma}


class RandomGamma(ImageOnlyTransform):
    """Performs the gamma transformation with a randomly chosen gamma. If image values (in any channel) are outside
        the [0,1] interval, this transformation is not performed.

        Args:
            gamma_limit (Tuple(float), optional): Interval from which gamma is selected.

                Defaults to ``(0.8, 1.2)``.
            always_apply (bool, optional): Always apply this transformation in composition. 
            
                Defaults to ``False``.
            p (float, optional): Chance of applying this transformation in composition. 
            
                Defaults to ``0.5``.

        Targets:
            image
    """
    def __init__(self, gamma_limit: Tuple[float] = (0.8, 1.2),
                 always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.gamma_limit = gamma_limit

    def apply(self, img, gamma=1, **params):
        return F.gamma_transform(img, gamma=gamma)

    def get_params(self, **data):
        return {"gamma": random.uniform(self.gamma_limit[0], self.gamma_limit[1])}

    def __repr__(self):
        return f'RandomGamma({self.gamma_limit}, {self.always_apply}, {self.p})'


class RandomBrightnessContrast(ImageOnlyTransform):
    """Randomly change brightness and contrast of the input image.

        Unlike ``RandomBrightnessContrast`` from `Albumentations`, this transform is using the
        formula :math:`f(a) = (c+1) * a + b`, where :math:`c` is contrast and :math:`b` is brightness.

        Args:
            brightness_limit ((float, float) | float, optional): Interval from which the change in brightness is
                randomly drawn. If the change in brightness is 0, the brightness will not change.

                Must be either of: ``B``, ``(B1, B2)``.

                If a float, the interval will be ``(-B, B)``.

                Defaults to ``0.2``.
            contrast_limit ((float, float) | float, optional): Interval from which the change in contrast is
                randomly drawn. If the change in contrast is 1, the contrast will not change.

                Must be either of: ``C``, ``(C1, C2)``.

                If a float, the interval will be ``(-C, C)``.

                Defaults to ``0.2``.
            always_apply (bool, optional): Always apply this transformation in composition. 
            
                Defaults to ``False``.
            p (float, optional): Chance of applying this transformation in composition. 
            
                Defaults to ``0.5``.

        Targets:
            image
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

    def __repr__(self):
        return f'RandomBrightnessContrast({self.brightness_limit}, {self.contrast_limit},  ' \
               f'{self.always_apply}, {self.p})'


class HistogramEqualization(ImageOnlyTransform):
    """Performs equalization of histogram. The equalization is done channel-wise, meaning that each channel is equalized
        separately.

        **Warning! Images are normalized over both spatial and temporal domains together. The output is in the range [0, 1].**

        Args:
            bins (int, optional): Number of bins for image histogram.

                Defaults to ``256``.
            always_apply (bool, optional): Always apply this transformation in composition. 
            
                Defaults to ``False``.
            p (float, optional): Chance of applying this transformation in composition. 
            
                Defaults to ``1``.

        Targets:
            image
    """
    def __init__(self, bins: int = 256, always_apply: bool = False, p: float = 1):
        super().__init__(always_apply, p)
        self.bins = bins

    def apply(self, img, **params):
        return F.histogram_equalization(img, self.bins)


class Pad(DualTransform):
    """Pads the input.

        Args:
            pad_size (int | Tuple[int] | List[int | Tuple[int]]): Number of pixels padded to the edges of each axis.

                Must be either of: ``P``, ``(P1, P2)``, ``[P_Z, P_Y, P_X]``, ``[P_Z, P_Y, P_X, P_T]``,
                ``[(P_Z1, P_Z2), (P_Y1, P_Y2), (P_X1, P_X2)]``, or
                ``[(P_Z1, P_Z2), (P_Y1, P_Y2), (P_X1, P_X2), (P_T1, P_T2)]``.

                If an integer, it is equivalent to ``[(P, P), (P, P), (P, P)]``.

                If a tuple, it is equivalent to ``[(P1, P2), (P1, P2), (P1, P2)]``.

                If a list, it must specify padding for all spatial dimensions and possibly also for the time dimension.

                The unspecified dimensions (C and possibly T) are not affected.
            border_mode (str, optional): Values outside image domain are filled according to this mode.

                Defaults to ``'constant'``.
            ival (float | Sequence, optional): Values of `image` voxels outside of the `image` domain.
                Only applied when ``border_mode = 'constant'`` or ``border_mode = 'linear_ramp'``.

                Defaults to ``0``.
            mval (float | Sequence, optional): Values of `mask` voxels outside of the `mask` domain.
                Only applied when ``border_mode = 'constant'`` or ``border_mode = 'linear_ramp'``.

                Defaults to ``0``.
            ignore_index (float | None, optional): If a float, then transformation of `mask` is done with 
                ``border_mode = 'constant'`` and ``mval = ignore_index``. 
                
                If ``None``, this argument is ignored.

                Defaults to ``None``.
            always_apply (bool, optional): Always apply this transformation in composition. 
            
                Defaults to ``True``.
            p (float, optional): Chance of applying this transformation in composition. 
            
                Defaults to ``1``.

        Targets:
            image, mask, float_mask
    """
    def __init__(self, pad_size: Union[int, Tuple[int],  List[Union[int, Tuple[int]]]], border_mode: str = 'constant',
                 ival: Union[float, Sequence] = 0, mval: Union[float, Sequence] = 0,
                 ignore_index: Union[float, None] = None, always_apply: bool = True, p : float = 1):
        super().__init__(always_apply, p)
        self.pad_size: TypeSextetInt = parse_pads(pad_size)
        self.border_mode = border_mode
        self.mask_mode = border_mode 
        self.ival = ival
        self.mval = mval

        if not (ignore_index is None):
            self.mask_mode = "constant"
            self.mval = ignore_index

    def apply(self, img, **params):
        return F.pad_pixels(img, self.pad_size, self.border_mode, self.ival)

    def apply_to_mask(self, mask, **params):
        return F.pad_pixels(mask, self.pad_size, self.mask_mode, self.mval, True)

    def apply_to_keypoints(self, keypoints, **params):
        return F.pad_keypoints(keypoints, self.pad_size)

    def __repr__(self):
        return f'Pad({self.pad_size}, {self.border_mode}, {self.ival}, {self.mval}, {self.always_apply}, ' \
               f'{self.p})'


class Normalize(ImageOnlyTransform):
    """Change image mean and standard deviation to the given values (channel-wise).

        Args:
            mean (float | List[float], optional): The desired channel-wise means.

                Must be either of: ``M``, ``[M_1, M_2, ..., M_C]``.

                Defaults to ``0``.
            std (float | List[float], optional): The desired channel-wise standard deviations.

                Must be either of: ``S``, ``[S_1, S_2, ..., S_C]``.

                Defaults to ``1``.
            always_apply (bool, optional): Always apply this transformation in composition. 
            
                Defaults to ``True``.
            p (float, optional): Chance of applying this transformation in composition. 
            
                Defaults to ``1``.

        Targets:
            image
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
    """Transform the image data to a contiguous array.

        Args:
            always_apply (bool, optional): Always apply this transformation in composition.

                Defaults to ``True``.
            p (float, optional): Chance of applying this transformation in composition.

                Defaults to ``1``.

        Targets:
            image, mask, float_mask
    """
    def __init__(self, always_apply: bool = True, p: float = 1.0):
        super().__init__(always_apply, p)

    def apply(self, image, **params):
        return np.ascontiguousarray(image)

    def apply_to_mask(self, mask, **params):
        return np.ascontiguousarray(mask)

    def __repr__(self):
        return f'Contiguous({self.always_apply}, {self.p})'


class StandardizeDatatype(DualTransform):
    """Change image and float_mask datatype to ``np.float32`` without changing intensities.
    Change mask datatype to ``np.int32``.

        Args:
            always_apply (bool, optional): Always apply this transformation in composition.

                Defaults to ``True``.
            p (float, optional): Chance of applying this transformation in composition.

                Defaults to ``1``.

        Targets:
            image, mask, float_mask
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
        return f'Float({self.always_apply}, {self.p})'



