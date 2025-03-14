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
#                Samuel Šuľan                                                                   #
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

from typing import List, Sequence, Tuple, Union, Optional
import numpy as np

from .utils import parse_limits, parse_coefs, parse_pads, to_tuple, get_spatio_temporal_domain_limit,\
    to_spatio_temporal, get_spatial_shape_from_image, get_sigma_axiswise
from ..core.transforms_interface import DualTransform, ImageOnlyTransform
from ..augmentations import functional as F
from ..augmentations.sitk_utils import parse_itk_interpolation
from ..biovol_typing import *
from ..random_utils import uniform, sample_range_uniform, randint, shuffle, sample


##########################################################################################
#                                                                                        #
#                                GEOMETRIC TRANSFORMATIONS                               #
#                                                                                        #
##########################################################################################

# TODO anti_aliasing_downsample keep parameter or remove?
class Resize(DualTransform):
    """Resize input to the given shape.

        Internally, the ``skimage.transform.resize`` function is used.
        The ``interpolation``, ``border_mode``, ``ival``, ``mval``,
        and ``anti_aliasing_downsample`` arguments are forwarded to it. More details at:
        https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize.

        Args:
            shape (tuple of ints): The desired image shape.

                Must be ``(Z, Y, X)``.

                The unspecified dimensions (C and T) are not affected.
            interpolation (int, optional): Order of spline interpolation.

                Defaults to ``1``.
            border_mode (str, optional): Values outside image domain are filled according to this mode.

                Defaults to ``'reflect'``.
            ival (float, optional): Value of `image` voxels outside of the `image` domain.
                Only applied when ``border_mode = 'constant'``.

                Defaults to ``0``.
            mval (float, optional): Value of `mask` and `float_mask` voxels outside of the domain.
                Only applied when ``border_mode = 'constant'``.
            
                Defaults to ``0``.
            anti_aliasing_downsample (bool, optional): Apply the Gaussian filter before downsampling. Recommended.
                
                Defaults to ``True``.
            ignore_index (float | None, optional): If specified, then transformation of `mask` is done with
                ``border_mode = 'constant'`` and ``mval = ignore_index``. 
                
                If ``None``, this argument is ignored.
                
                Defaults to ``None``.
            always_apply (bool, optional): Always apply this transformation in composition. 
            
                Defaults to ``False``.
            p (float, optional): Probability of applying this transformation in composition.
            
                Defaults to ``1``.

        Targets:
            image, mask, float mask, key points, bounding boxes
    """
    def __init__(self, shape: TypeSpatialShape, interpolation: int = 1, border_mode: str = 'reflect', ival: float = 0,
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
            self.mask_mode = 'constant'
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

    def get_params(self, targets, **data):

        # read shape of the original image
        domain_limit: TypeSpatioTemporalCoordinate = get_spatio_temporal_domain_limit(data, targets)

        return {
            'domain_limit': domain_limit,
        }
        
    def __repr__(self):
        return f'Resize(shape={self.shape}, interpolation={self.interpolation}, border_mode={self.border_mode}, ' \
               f'ival={self.ival}, mval={self.mval}, anti_aliasing_downsample={self.anti_aliasing_downsample}, ' \
               f'always_apply={self.always_apply}, p={self.p})'


class Rescale(DualTransform):
    """ Rescale the input and change its shape accordingly.

        Internally, the ``skimage.transform.resize`` function is used.
        The ``interpolation``, ``border_mode``, ``ival``, ``mval``,
        and ``anti_aliasing_downsample`` arguments are forwarded to it. More details at:
        https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize.

        Args:
            scales (float|List[float], optional): Value by which the input should be scaled.

                Must be either of: ``S``, ``[S_Z, S_Y, S_X]``.

                If a float, then all spatial dimensions are scaled by it (equivalent to ``[S, S, S]``).

                The unspecified dimensions (C and T) are not affected.

                Defaults to ``1``.
            interpolation (int, optional): Order of spline interpolation.

                Defaults to ``1``.
            border_mode (str, optional): Values outside image domain are filled according to this mode.

                Defaults to ``'reflect'``.
            ival (float, optional): Value of `image` voxels outside of the `image` domain.
                Only applied when ``border_mode = 'constant'``.

                Defaults to ``0``.
            mval (float, optional): Value of `mask` and `float_mask` voxels outside of the domain.
                Only applied when ``border_mode = 'constant'``.

                Defaults to ``0``.
            anti_aliasing_downsample (bool, optional): Apply the Gaussian filter before downsampling. Recommended.

                Defaults to ``True``.
            ignore_index (float | None, optional): If specified, then transformation of `mask` is done with
                ``border_mode = 'constant'`` and ``mval = ignore_index``.

                If ``None``, this argument is ignored.

                Defaults to ``None``.
            always_apply (bool, optional): Always apply this transformation in composition.

                Defaults to ``True``.
            p (float, optional): Probability of applying this transformation in composition.

                Defaults to ``1``.

        Targets:
            image, mask, float mask, key points, bounding boxes
        """

    def __init__(self, scales=1, interpolation: int = 1, border_mode: str = 'reflect', ival: float = 0,
                 mval: float = 0, anti_aliasing_downsample: bool = True, ignore_index=None,
                 always_apply: bool = True, p: float = 1, **kwargs):
        super().__init__(always_apply, p)
        self.scale = parse_coefs(scales, identity_element=1.)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.mask_mode = border_mode
        self.ival = ival
        self.mval = mval
        self.anti_aliasing_downsample = anti_aliasing_downsample
        if not (ignore_index is None):
            self.mask_mode = 'constant'
            self.mval = ignore_index

    def apply(self, img, **params):
        return F.resize(img, input_new_shape=params['new_shape'], interpolation=self.interpolation, cval=self.ival,
                        border_mode=self.border_mode, anti_aliasing_downsample=self.anti_aliasing_downsample)

    def apply_to_mask(self, mask, **params):
        return F.resize(mask, input_new_shape=params['new_shape'], interpolation=0, cval=self.mval,
                        border_mode=self.mask_mode, anti_aliasing_downsample=False, mask=True)

    def apply_to_float_mask(self, mask, **params):
        return F.resize(mask, input_new_shape=params['new_shape'], interpolation=self.interpolation, cval=self.mval,
                        border_mode=self.mask_mode, anti_aliasing_downsample=False, mask=True)

    def apply_to_keypoints(self, keypoints, **params):
        return F.resize_keypoints(keypoints,
                                  domain_limit=params['domain_limit'],
                                  new_shape=params['new_shape'])

    """
    def apply_to_bboxes(self, bboxes, **params):
        for bbox in bboxes:
            new_bbox = F.resize_keypoints(bbox,
                                          input_new_shape=params['new_shape'],
                                          original_shape=params['original_shape'],
                                          keep_all=True)

            if validate_bbox(bbox, new_bbox, min_overlay_ratio):
                res.append(new_bbox)

        return res
    """

    def get_params(self, targets, **data):
        # read shape of the original image
        domain_limit: TypeSpatioTemporalCoordinate = get_spatio_temporal_domain_limit(data, targets)

        # compute shape of the resize dimage
        # TODO +(0,) because of the F.resize error/hotfix
        new_shape = tuple(np.asarray(domain_limit[:3]) * np.asarray(self.scale)) + (0,)

        return {
            'domain_limit': domain_limit,
            'new_shape': new_shape,
        }

    def __repr__(self):
        return f'Rescale(scales={self.scale}, interpolation={self.interpolation}, border_mode={self.border_mode}, ' \
               f'ival={self.ival}, mval={self.mval}, anti_aliasing_downsample={self.anti_aliasing_downsample}, ' \
               f'always_apply={self.always_apply}, p={self.p})'


class Scale(DualTransform):
    """Rescale the input image content by the given scale. The image shape remains unchanged.

        Args:
            scales (float|List[float], optional): Value by which the input should be scaled.

                Must be either of: ``S``, ``[S_Z, S_Y, S_X]``.

                If a float, then all spatial dimensions are scaled by it (equivalent to ``[S, S, S]``).

                The unspecified dimensions (C and T) are not affected.

                Defaults to ``1``.
            interpolation (str, optional): SimpleITK interpolation type for `image` and `float_mask`.

                Must be one of ``linear``, ``nearest``, ``bspline``, ``gaussian``.

                For `mask`, the ``nearest`` interpolation is always used.

                Defaults to ``linear``.
            spacing (float | Tuple[float, float, float] | None, optional): Voxel spacing for individual
                spatial dimensions.

                Must be either of: ``S``, ``(S1, S2, S3)``, or ``None``.

                If ``None``, equivalent to ``(1, 1, 1)``.

                If a float ``S``, equivalent to ``(S, S, S)``.

                Otherwise, a scale for each spatial dimension must be given.

                Defaults to ``None``.
            border_mode (str, optional): Values outside image domain are filled according to this mode.

                Defaults to ``'constant'``.
            ival (float, optional): Value of `image` voxels outside of the `image` domain.
                Only applied when ``border_mode = 'constant'``.

                Defaults to ``0``.
            mval (float, optional): Value of `mask` and `float_mask` voxels outside of the domain.
                Only applied when ``border_mode = 'constant'``.

                Defaults to ``0``.
            ignore_index (float | None, optional): If specified, then transformation of `mask` is done with
                ``border_mode = 'constant'`` and ``mval = ignore_index``. 
                
                If ``None``, this argument is ignored.

                Defaults to ``None``.
            always_apply (bool, optional): Always apply this transformation in composition. 
            
                Defaults to ``False``.
            p (float, optional): Probability of applying this transformation in composition.
            
                Defaults to ``1``.

        Targets:
            image, mask, float mask, key points, bounding boxes
    """
    def __init__(self, scales: Union[float, TypeTripletFloat] = 1,
                 interpolation: str = 'linear', spacing: Union[float, TypeTripletFloat] = None,
                 border_mode: str = 'constant', ival: float = 0, mval: float = 0,
                 ignore_index: Union[float, None] = None, always_apply: bool = False, p: float = 1):
        super().__init__(always_apply, p)
        self.scale = parse_coefs(scales, identity_element=1.)
        self.interpolation: str = parse_itk_interpolation(interpolation)
        self.spacing: TypeTripletFloat = parse_coefs(spacing, identity_element=1.)
        self.border_mode = border_mode              # not implemented
        self.mask_mode = border_mode                # not implemented
        self.ival = ival
        self.mval = mval
        if not (ignore_index is None):
            self.mask_mode = 'constant'
            self.mval = ignore_index

    def apply(self, img, **params):
        return F.affine(img,
                        scales=self.scale,
                        interpolation=self.interpolation,
                        border_mode=self.border_mode,
                        value=self.ival,
                        spacing=self.spacing)

    def apply_to_mask(self, mask, **params):
        interpolation = parse_itk_interpolation('nearest')   # refers to 'sitkNearestNeighbor'
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

    def get_params(self, targets, **data):
        domain_limit: TypeSpatioTemporalCoordinate = get_spatio_temporal_domain_limit(data, targets)
        return {'domain_limit': domain_limit}

    def __repr__(self):
        return f'Scale(scales={self.scale}, interpolation={self.interpolation}, spacing={self.spacing}, ' \
               f'border_mode={self.border_mode}, ival={self.ival}, mval={self.mval},' \
               f'always_apply={self.always_apply}, p={self.p})'


class RandomScale(DualTransform):
    """Randomly rescale the input image content by the given scale. The image shape remains unchanged.

        Args:
            scaling_limit (float | Tuple[float], optional): Limits of scaling factors.

                Must be either of: ``S``, ``(S1, S2)``, ``(S_Z, S_Y, S_X)``,
                or ``(S_Z1, S_Z2, S_Y1, S_Y2, S_X1, S_X2)``.

                If a float ``S``, then all spatial dimensions are scaled by a random number drawn uniformly from
                the interval [1/S, S] (equivalent to inputting ``(1/S, S, 1/S, S, 1/S, S)``).

                If a tuple of 2 floats, then all spatial dimensions are scaled by a random number drawn uniformly
                from the interval [S1, S2] (equivalent to inputting ``(S1, S2, S1, S2, S1, S2)``).

                If a tuple of 3 floats, then an interval [1/S_a, S_a] is constructed for each spatial
                dimension and the scale is randomly drawn from it
                (equivalent to inputting ``(1/S_Z, S_Z, 1/S_Y, S_Y, 1/S_X, S_X)``).

                If a tuple of 6 floats, the scales for individual spatial dimensions are randomly drawn from the
                respective intervals [S_Z1, S_Z2], [S_Y1, S_Y2], [S_X1, S_X2].

                The unspecified dimensions (C and T) are not affected.

                Defaults to ``(1.1)``.

            interpolation (str, optional): SimpleITK interpolation type for `image` and `float_mask`.

                Must be one of ``linear``, ``nearest``, ``bspline``, ``gaussian``.

                For `mask`, the ``nearest`` interpolation is always used.

                Defaults to ``linear``.

            spacing (float | Tuple[float, float, float] | None, optional): Voxel spacing for individual spatial
                dimensions.

                Must be either of: ``S``, ``(S1, S2, S3)``, or ``None``.

                If ``None``, equivalent to ``(1, 1, 1)``.

                If a float ``S``, equivalent to ``(S, S, S)``.

                Otherwise, a scale for each spatial dimension must be given.

                Defaults to ``None``.

            border_mode (str, optional): Values outside image domain are filled according to the mode.

                Defaults to ``'constant'``.

            ival (float, optional): Value of `image` voxels outside of the `image` domain.
                Only applied when ``border_mode = 'constant'``.

                Defaults to ``0``.

            mval (float, optional): Value of `mask` and `float_mask` voxels outside of the domain.
                Only applied when ``border_mode = 'constant'``.

                Defaults to ``0``.

            ignore_index (float | None, optional): If specified, then transformation of `mask` is done with
                ``border_mode = 'constant'`` and ``mval = ignore_index``. 
                
                If ``None``, this argument is ignored.

                Defaults to ``None``.

            always_apply (bool, optional): Always apply this transformation in composition.

                Defaults to ``False``.

            p (float, optional): Probability of applying this transformation in composition.

                Defaults to ``0.5``.

        Targets:
            image, mask, float mask, key points, bounding boxes
    """      
    def __init__(self, scaling_limit: Union[float, TypePairFloat, TypeTripletFloat, TypeSextetFloat] = (0.9, 1.1),
                 interpolation: str = 'linear', spacing: Union[float, TypeTripletFloat] = None,
                 border_mode: str = 'constant', ival: float = 0, mval: float = 0,
                 ignore_index: Union[float, None] = None, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.scaling_limit: TypeSextetFloat = parse_limits(scaling_limit, scale=True)
        self.interpolation: str = parse_itk_interpolation(interpolation)
        self.spacing: TypeTripletFloat = parse_coefs(spacing, identity_element=1.)
        self.border_mode = border_mode
        self.mask_mode = border_mode
        self.ival: float = ival
        self.mval: float = mval
        if not (ignore_index is None):
            self.mask_mode = 'constant'
            self.mval = ignore_index

    def get_params(self, targets, **data):
        # set parameters of the transform
        domain_limit: TypeSpatioTemporalCoordinate = get_spatio_temporal_domain_limit(data, targets)
        scale = sample_range_uniform(self.scaling_limit)

        return {
            'domain_limit': domain_limit,
            'scale': scale,
        }

    def apply(self, img, **params):
        return F.affine(img,
                        scales=params['scale'],
                        interpolation=self.interpolation,
                        border_mode=self.border_mode,
                        value=self.ival,
                        spacing=self.spacing)

    def apply_to_mask(self, mask, **params):
        interpolation = parse_itk_interpolation('nearest')   # refers to 'sitkNearestNeighbor'
        return F.affine(np.expand_dims(mask, 0),
                        scales=params['scale'],
                        interpolation=interpolation,
                        border_mode=self.mask_mode,
                        value=self.mval,
                        spacing=self.spacing)[0]

    def apply_to_float_mask(self, mask, **params):
        return F.affine(np.expand_dims(mask, 0),
                        scales=params['scale'],
                        interpolation=self.interpolation,
                        border_mode=self.mask_mode,
                        value=self.mval,
                        spacing=self.spacing)[0]

    def apply_to_keypoints(self, keypoints, **params):
        return F.affine_keypoints(keypoints,
                                  scales=params['scale'],
                                  spacing=self.spacing,
                                  domain_limit=params['domain_limit'])

    def __repr__(self):
        return f'RandomScale(scaling_limit={self.scaling_limit}, interpolation={self.interpolation}, ' \
               f'spacing={self.spacing}, border_mode={self.border_mode}, ival={self.ival}, mval={self.mval}, ' \
               f'always_apply={self.always_apply}, p={self.p})'


class AffineTransform(DualTransform):
    """Affine transformation of the input image with given parameters. Image shape remains unchanged.

        Args:
            angles (Tuple[float], optional): Angles of rotation (in degrees) for the spatial axes.

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
            interpolation (str, optional): SimpleITK interpolation type for `image` and `float_mask`.

                Must be one of ``linear``, ``nearest``, ``bspline``, ``gaussian``.

                For `mask`, the ``nearest`` interpolation is always used.

                Defaults to ``linear``.
            border_mode (str, optional): Values outside image domain are filled according to this mode.

                Defaults to ``'constant'``.
            ival (float, optional): Value of `image` voxels outside of the `image` domain.
                Only applied when ``border_mode = 'constant'``.

                Defaults to ``0``.
            mval (float, optional): Value of `mask` and `float_mask` voxels outside of the domain.
                Only applied when ``border_mode = 'constant'``.

                Defaults to ``0``.
            ignore_index (float | None, optional): If specified, then transformation of `mask` is done with
                ``border_mode = 'constant'`` and ``mval = ignore_index``.

                If ``None``, this argument is ignored.

                Defaults to ``None``.
            always_apply (bool, optional): Always apply this transformation in composition.

                Defaults to ``False``.
            p (float, optional): Probability of applying this transformation in composition.

                Defaults to ``0.5``.

        Targets:
            image, mask, float mask, key points, bounding boxes
    """

    def __init__(self, angles: TypeTripletFloat = (0, 0, 0),
                 translation: TypeTripletFloat = (0, 0, 0),
                 scale: TypeTripletFloat = (1, 1, 1),
                 spacing: TypeTripletFloat = (1, 1, 1),
                 change_to_isotropic: bool = False,
                 interpolation: str = 'linear',
                 border_mode: str = 'constant', ival: float = 0, mval: float = 0,
                 ignore_index: Union[float, None] = None, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.angles: TypeTripletFloat = parse_coefs(angles, identity_element=0)
        self.translation: TypeTripletFloat = parse_coefs(translation, identity_element=0)
        self.scale: TypeTripletFloat = parse_coefs(scale, identity_element=1)
        self.spacing: TypeTripletFloat = parse_coefs(spacing, identity_element=1)
        self.interpolation: str = parse_itk_interpolation(interpolation)
        self.border_mode = border_mode  # not used
        self.mask_mode = border_mode  # not used
        self.ival = ival
        self.mval = mval
        self.keep_scale = not change_to_isotropic

        if ignore_index is not None:
            self.mask_mode = 'constant'
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
        interpolation = parse_itk_interpolation('nearest')  # refers to 'sitkNearestNeighbor'
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

    def get_params(self, targets, **data):
        # set parameters of the transform
        domain_limit = get_spatio_temporal_domain_limit(data, targets)

        return {
            'domain_limit': domain_limit
        }

    def __repr__(self):
        return f'AffineTransform(angles={self.angles}, translation={self.translation}, scale={self.scale}, ' \
               f'spacing={self.spacing}, change_to_isotropic={not self.keep_scale}, ' \
               f'interpolation={self.interpolation}, border_mode={self.border_mode}, ival={self.ival}, ' \
               f'mval={self.mval}, always_apply={self.always_apply}, p={self.p})'


class RandomAffineTransform(DualTransform):
    """Affine transformation of the input image with randomly chosen parameters. Image shape remains unchanged.

        Args:
            angle_limit (Tuple[float] | float, optional): Intervals (in degrees) from which angles of
                rotation for the spatial axes are chosen.

                Must be either of: ``A``, ``(A1, A2)``, ``(A1, A2, A3)``, or ``(A_Z1, A_Z2, A_Y1, A_Y2, A_X1, A_X2)``.

                If a float, equivalent to ``(-A, A, -A, A, -A, A)``.

                If a tuple with 2 items, equivalent to ``(A1, A2, A1, A2, A1, A2)``.

                If a tuple with 3 items, equivalent to ``(-A1, A1, -A2, A2, -A3, A3)``.

                If a tuple with 6 items, angle of rotation is randomly chosen from an interval [A_a1, A_a2] for each
                spatial axis.

                Defaults to ``(15, 15, 15)``.
            translation_limit (Tuple[float] | float | None, optional): Intervals from which the translation parameters
                for the spatial axes are chosen.

                Must be either of: ``T``, ``(T1, T2)``, ``(T1, T2, T3)``, or ``(T_Z1, T_Z2, T_Y1, T_Y2, T_X1, T_X2)``.

                If a float, equivalent to ``(2-T, T, 2-T, T, 2-T, T)``.

                If a tuple with 2 items, equivalent to ``(T1, T2, T1, T2, T1, T2)``.

                If a tuple with 3 items, equivalent to ``(2-T1, T1, 2-T2, T2, 2-T3, T3)``.

                If a tuple with 6 items, the translation parameter is randomly chosen from an interval [T_a1, T_a2] for
                each spatial axis.

                Defaults to ``(0, 0, 0)``.
            scaling_limit (Tuple[float] | float, optional): Intervals from which the scales for the spatial axes
                are chosen.

                Must be either of: ``S``, ``(S1, S2)``, ``(S1, S2, S3)``, or ``(S_Z1, S_Z2, S_Y1, S_Y2, S_X1, S_X2)``.

                If a float, equivalent to ``(1/S, S, 1/S, S, 1/S, S)``.

                If a tuple with 2 items, equivalent to ``(S1, S2, S1, S2, S1, S2)``.

                If a tuple with 3 items, equivalent to ``(1/S1, S1, 1/S2, S2, 1/S3, S3)``.

                If a tuple with 6 items, the scale is randomly chosen from an interval [S_a1, S_a2] for
                each spatial axis.

                Defaults to ``(1., 1., 1.)``.
            spacing (float | Tuple[float, float, float] | None, optional): Voxel spacing for individual spatial
                dimensions.

                Must be either of: ``S``, ``(S1, S2, S3)``, or ``None``.

                If ``None``, equivalent to ``(1, 1, 1)``.

                If a float ``S``, equivalent to ``(S, S, S)``.

                Otherwise, a scale for each spatial dimension must be given.

                Defaults to ``None``.
            change_to_isotropic (bool, optional): Change data from anisotropic to isotropic.

                Defaults to ``False``.
            interpolation (str, optional): SimpleITK interpolation type for `image` and `float_mask`.

                Must be one of ``linear``, ``nearest``, ``bspline``, ``gaussian``.

                For `mask`, the ``nearest`` interpolation is always used.

                Defaults to ``linear``.
            border_mode (str, optional): Values outside image domain are filled according to this mode.

                Defaults to ``'constant'``.
            ival (float, optional): Value of `image` voxels outside of the `image` domain.
                Only applied when ``border_mode = 'constant'``.

                Defaults to ``0``.
            mval (float, optional): Value of `mask` and `float_mask` voxels outside of the domain.
                Only applied when ``border_mode = 'constant'``.

                Defaults to ``0``.
            ignore_index (float | None, optional): If specified, then transformation of `mask` is done with
                ``border_mode = 'constant'`` and ``mval = ignore_index``.

                If ``None``, this argument is ignored.

                Defaults to ``None``.
            always_apply (bool, optional): Always apply this transformation in composition.

                Defaults to ``False``.
            p (float, optional): Probability of applying this transformation in composition.

                Defaults to ``0.5``.

        Targets:
            image, mask, float mask, key points, bounding boxes
    """

    def __init__(self, angle_limit: Union[float, TypePairFloat, TypeTripletFloat, TypeSextetFloat] = (15., 15., 15.),
                 translation_limit: Union[float, TypePairFloat, TypeTripletFloat, TypeSextetFloat] = (0., 0., 0.),
                 scaling_limit: Union[float, TypePairFloat, TypeTripletFloat, TypeSextetFloat] = (1., 1., 1.),
                 spacing: Union[float, TypeTripletFloat] = None,
                 change_to_isotropic: bool = False,
                 interpolation: str = 'linear',
                 border_mode: str = 'constant', ival: float = 0, mval: float = 0,
                 ignore_index: Union[float, None] = None, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.angle_limit: TypeSextetFloat = parse_limits(angle_limit)
        self.translation_limit: TypeSextetFloat = parse_limits(translation_limit)
        self.scaling_limit: TypeSextetFloat = parse_limits(scaling_limit, scale=True)
        self.spacing: TypeTripletFloat = parse_coefs(spacing, identity_element=1)
        self.interpolation: int = parse_itk_interpolation(interpolation)
        self.border_mode = border_mode  # not used
        self.mask_mode = border_mode  # not used
        self.ival = ival
        self.mval = mval
        self.keep_scale = not change_to_isotropic

        if ignore_index is not None:
            self.mask_mode = 'constant'
            self.mval = ignore_index

    def apply(self, img, **params):
        return F.affine(img,
                        scales=params['scale'],
                        degrees=params['angles'],
                        translation=params['translation'],
                        interpolation=self.interpolation,
                        border_mode=self.border_mode,
                        value=self.ival,
                        spacing=self.spacing)

    def apply_to_mask(self, mask, **params):
        interpolation = parse_itk_interpolation('nearest')  # refers to 'sitkNearestNeighbor'
        return F.affine(np.expand_dims(mask, 0),
                        scales=params['scale'],
                        degrees=params['angles'],
                        translation=params['translation'],
                        interpolation=interpolation,
                        border_mode=self.mask_mode,
                        value=self.mval,
                        spacing=self.spacing)[0]

    def apply_to_float_mask(self, mask, **params):
        return F.affine(np.expand_dims(mask, 0),
                        scales=params['scale'],
                        degrees=params['angles'],
                        translation=params['translation'],
                        interpolation=self.interpolation,
                        border_mode=self.mask_mode,
                        value=self.mval,
                        spacing=self.spacing)[0]

    def apply_to_keypoints(self, keypoints, **params):
        return F.affine_keypoints(keypoints,
                                  scales=params['scale'],
                                  degrees=params['angles'],
                                  translation=params['translation'],
                                  spacing=self.spacing,
                                  domain_limit=params['domain_limit'])

    def get_params(self, targets, **data):
        # set parameters of the transform
        scales = sample_range_uniform(self.scaling_limit)
        angles = sample_range_uniform(self.angle_limit)
        translation = sample_range_uniform(self.translation_limit)
        domain_limit = get_spatio_temporal_domain_limit(data, targets)

        return {
            'scale': scales,
            'angles': angles,
            'translation': translation,
            'domain_limit': domain_limit
        }

    def __repr__(self):
        return f'RandomAffineTransform(angle_limit={self.angle_limit}, translation_limit={self.translation_limit}, ' \
               f'scaling_limit={self.scaling_limit}, spacing={self.spacing}, ' \
               f'change_to_isotropic={not self.keep_scale}, interpolation={self.interpolation}, ' \
               f'border_mode={self.border_mode}, ival={self.ival}, mval={self.mval}, ' \
               f'always_apply={self.always_apply}, p={self.p})'


class RandomRotate90(DualTransform):
    """Rotation of input by 0, 90, 180, or 270 degrees around the specified spatial axes.

        Args:
            axes (List[int], optional): List of axes around which the input is rotated. Recognised axis symbols are
                ``1`` for Z, ``2`` for Y, and ``3`` for X. A single axis can occur multiple times in the list.
                If ``shuffle_axis = False``, the order of axes determines the order of transformations.
                If ``None``, will be rotated around all spatial axes.

                Defaults to ``None``.
            shuffle_axis (bool, optional): If set to ``True``, the order of rotations is random.

                Defaults to ``False``.
            factor (int, optional): The number of times the array is rotated by 90 degrees.
                If ``None``, will be chosen randomly.

                Defaults to ``None``.
            always_apply (bool, optional): Always apply this transformation in composition. 
            
                Defaults to ``False``.
            p (float, optional): Probability of applying this transformation in composition.
            
                Defaults to ``0.5``.

        Targets:
            image, mask, float mask, key points, bounding boxes
    """
    def __init__(self, axes: List[int] = None, shuffle_axis: bool = False, factor: Optional[int] = None,
                 always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.axes = axes
        self.shuffle_axis = shuffle_axis
        self.factor = factor

    def apply(self, img, **params):
        for factor, axes in zip(params['factor'], params['rotation_around']):
            img = np.rot90(img, factor, axes=axes)
        return img

    def apply_to_mask(self, mask, **params):
        for rot, factor in zip(params['rotation_around'], params['factor']):
            mask = np.rot90(mask, factor, axes=(rot[0] - 1, rot[1] - 1))
        return mask

    def apply_to_keypoints(self, keypoints, **params):
        for rot, factor in zip(params['rotation_around'], params['factor']):
            keypoints = F.rot90_keypoints(keypoints,
                                          factor=factor,
                                          axes=(rot[0], rot[1]),
                                          img_shape=params['img_shape'])
        return keypoints

    def get_params(self, targets, **data):

        # Rotate around all spatial axes if not specified by the user:
        if self.axes is None:
            self.axes = [1, 2, 3]

        # Create all combinations for rotating
        axes_to_rotate = {1: (2, 3), 2: (1, 3), 3: (1, 2)}
        rotation_around = []
        for i in self.axes:
            if i in axes_to_rotate.keys():
                rotation_around.append(axes_to_rotate[i])

        # Shuffle the order of rotation axes
        if self.shuffle_axis:
            shuffle(rotation_around)

        # If not specified, choose the angle to rotate
        if self.factor is None:
            factor = list(randint(0, 3, size=len(rotation_around)))
        else:
            factor = [self.factor]
            rotation_around = [(1, 2)]
            print('ROT90', factor, rotation_around)

        img_shape = get_spatial_shape_from_image(data, targets)

        return {'factor': factor,
                'rotation_around': rotation_around,
                'img_shape': img_shape}

    def __repr__(self):
        return f'RandomRotate90(axes={self.axes}, shuffle_axis={self.shuffle_axis}, factor={self.factor}, ' \
               f'always_apply={self.always_apply}, p={self.p})'


class Flip(DualTransform):
    """Flip input around the specified spatial axes.

        Args:
            axes (List[int], optional): List of axes around which is flip done. Recognised axis symbols are
                ``1`` for Z, ``2`` for Y, and ``3`` for X. If ``None``, will be flipped around all spatial axes.

                Defaults to ``None``.
            always_apply (bool, optional): Always apply this transformation in composition. 
            
                Defaults to ``False``.
            p (float, optional): Probability of applying this transformation in composition.
            
                Defaults to ``1``.

        Targets:
            image, mask, float mask, key points, bounding boxes
    """
    def __init__(self, axes: List[int] = None, always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.axes = axes

    def apply(self, img, **params):
        return np.flip(img, params['axes'])

    def apply_to_mask(self, mask, **params):
        # Mask has no dimension channel
        return np.flip(mask, axis=[item - 1 for item in params['axes']])

    def apply_to_keypoints(self, keypoints, **params):
        return F.flip_keypoints(keypoints,
                                axes=params['axes'],
                                img_shape=params['img_shape'])

    def get_params(self, targets, **data):
        # Use all spatial axes if not specified otherwise:
        axes = [1, 2, 3] if self.axes is None else self.axes
        # Get image shape (needed for keypoints):
        img_shape = get_spatial_shape_from_image(data, targets)

        return {'axes': axes,
                'img_shape': img_shape}

    def __repr__(self):
        return f'Flip(axes={self.axes}, always_apply={self.always_apply}, p={self.p})'


class RandomFlip(DualTransform):
    """Flip input around a set of axes randomly chosen from the input list of axis combinations.

        Args:
            axes_to_choose (List[int], Tuple[int], or None, optional): List of axis indices from which some are randomly
                chosen. Recognised axis symbols are ``1`` for Z, ``2`` for Y, and ``3`` for X. The image will be
                flipped around the chosen axes.

                If ``None``, a random subset of spatial axes is chosen, corresponding to inputting
                ``[1, 2, 3]``.

                Defaults to ``None``.
            always_apply (bool, optional): Always apply this transformation in composition. 
            
                Defaults to ``False``.
            p (float, optional): Probability of applying this transformation in composition.
            
                Defaults to ``0.5``.

        Targets:
            image, mask, float mask, key points, bounding boxes
    """
    def __init__(self, axes_to_choose: Union[None, List[int], Tuple[int]] = None, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.axes = axes_to_choose

    def apply(self, img, **params):
        return np.flip(img, params['axes'])

    def apply_to_mask(self, mask, **params):
        # Mask has no dimension channel
        return np.flip(mask, params['axes'] - 1)  # params['axes'] is a np.ndarray

    def apply_to_keypoints(self, keypoints, keep_all=False, **params):
        return F.flip_keypoints(keypoints,
                                axes=params['axes'],
                                img_shape=params['img_shape'])

    def get_params(self, targets, **data):
        if self.axes == []:
            axes = np.asarray(self.axes)
        else:
            # Use all spatial axes if not specified otherwise:
            to_choose = [1, 2, 3] if self.axes is None else self.axes
            # Randomly choose some axes from the given list:
            axes = sample(population=to_choose, k=randint(0, len(to_choose)))

        # Get image shape (needed for keypoints):
        img_shape = get_spatial_shape_from_image(data, targets)

        return {'axes': axes,
                'img_shape': img_shape}

    def __repr__(self):
        return f'RandomFlip(axes_to_choose={self.axes}, always_apply={self.always_apply}, p={self.p})'


class CenterCrop(DualTransform):
    """Crop the central region of the given size from the input .
          
        Unlike ``CenterCrop`` from `Albumentations`, this transform pads the input in dimensions
        where the input is smaller than the ``shape`` with ``numpy.pad``. The ``border_mode``, ``ival`` and ``mval``
        arguments are forwarded to ``numpy.pad`` if padding is necessary. More details at:
        https://numpy.org/doc/stable/reference/generated/numpy.pad.html.

        Args:
            shape (Tuple[int]): The desired shape of input.

                Must be ``[Z, Y, X]``.
            border_mode (str, optional): Values outside image domain are filled according to this mode.

                Defaults to ``'reflect'``.
            ival (float | Sequence, optional): Values of `image` voxels outside of the `image` domain.
                Only applied when ``border_mode = 'constant'`` or ``border_mode = 'linear_ramp'``.

                Defaults to ``(0, 0)``.
            mval (float | Sequence, optional): Values of `mask` voxels outside of the `mask` domain.
                Only applied when ``border_mode = 'constant'`` or ``border_mode = 'linear_ramp'``.

                Defaults to ``(0, 0)``.
            ignore_index (float | None, optional): If specified, then transformation of `mask` is done with
                ``border_mode = 'constant'`` and ``mval = ignore_index``. 
                
                If ``None``, this argument is ignored.

                Defaults to ``None``.
            always_apply (bool, optional): Always apply this transformation in composition. 
            
                Defaults to ``False``.
            p (float, optional): Probability of applying this transformation in composition.
            
                Defaults to ``1``.

        Targets:
            image, mask, float mask, key points, bounding boxes
    """
    def __init__(self, shape: TypeSpatialShape, border_mode: str = 'reflect',
                 ival: Union[Sequence[float], float] = (0, 0),
                 mval: Union[Sequence[float], float] = (0, 0), ignore_index: Union[float, None] = None,
                 always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply, p)
        self.output_shape = np.asarray(shape, dtype=np.intc)  # TODO: make it len 3 and type tuple
        self.border_mode = border_mode
        self.mask_mode = border_mode
        self.ival = ival
        self.mval = mval
        
        if not (ignore_index is None):
            self.mask_mode = 'constant'
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

    def get_params(self, targets, **data):
        # Get crop coordinates:
        # 1. Original image shape
        img_spatial_shape = get_spatial_shape_from_image(data, targets)
        # 2. Position of the corner closest to the image origin when cropping from the center of the image
        position: TypeSpatialCoordinate = (img_spatial_shape - self.output_shape) // 2
        position = np.maximum(position, 0).astype(int)
        # 3. Padding size if necessary
        pad_dims = F.get_pad_dims(img_spatial_shape, self.output_shape)

        return {'crop_position': position,
                'pad_dims': pad_dims}

    def __repr__(self):
        return f'CenterCrop(shape={self.output_shape}, border_mode={self.border_mode}, ival={self.ival}, ' \
               f'mval={self.mval}, always_apply={self.always_apply}, p={self.p})'


class RandomCrop(DualTransform):
    """Randomly crop a region of the given size from the input.

        Unlike ``RandomCrop`` from `Albumentations`, this transform pads the input in dimensions
        where the input is smaller than the ``shape`` with ``numpy.pad``. The ``border_mode``, ``ival`` and ``mval``
        arguments are forwarded to ``numpy.pad`` if padding is necessary. More details at:
        https://numpy.org/doc/stable/reference/generated/numpy.pad.html.

        Args:
            shape (Tuple[int]): The desired shape of input.

                Must be ``[Z, Y, X]``.
            border_mode (str, optional): Values outside image domain are filled according to this mode.

                Defaults to ``'reflect'``.
            ival (float | Sequence, optional): Values of `image` voxels outside of the `image` domain.
                Only applied when ``border_mode = 'constant'`` or ``border_mode = 'linear_ramp'``.

                Defaults to ``(0, 0)``.
            mval (float | Sequence, optional): Values of `mask` voxels outside of the `mask` domain.
                Only applied when ``border_mode = 'constant'`` or ``border_mode = 'linear_ramp'``.

                Defaults to ``(0, 0)``.
            ignore_index (float | None, optional): If specified, then transformation of `mask` is done with
                ``border_mode = 'constant'`` and ``mval = ignore_index``. 
                
                If ``None``, this argument is ignored.

                Defaults to ``None``.
            always_apply (bool, optional): Always apply this transformation in composition. 
            
                Defaults to ``False``.
            p (float, optional): Probability of applying this transformation in composition.
            
                Defaults to ``1``.

        Targets:
            image, mask, float mask, key points, bounding boxes
    """
    def __init__(self, shape: TypeSpatialShape, border_mode: str = 'reflect', ival: Union[Sequence[float], float] = (0, 0),
                 mval: Union[Sequence[float], float] = (0, 0), ignore_index: Union[float, None] = None,
                 always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply, p)
        self.output_shape = np.asarray(shape, dtype=np.intc)
        self.border_mode = border_mode
        self.mask_mode = border_mode
        self.ival = ival
        self.mval = mval

        if not (ignore_index is None):
            self.mask_mode = 'constant'
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

    def get_params(self, targets, **data):
        # Get crop coordinates:
        # 1. Original image shape
        img_spatial_shape = get_spatial_shape_from_image(data, targets)
        # 2. Position of the corner closest to the image origin, positioned randomly so that the whole crop is
        # within the image domain if possible
        ranges: TypeSpatialShape = np.maximum(img_spatial_shape - self.output_shape, 0)
        position = randint(0, ranges)
        # 3. Padding size if necessary
        pad_dims = F.get_pad_dims(img_spatial_shape, self.output_shape)

        return {'crop_position': position,
                'pad_dims': pad_dims}

    def __repr__(self):
        return f'RandomCrop(shape={self.output_shape}, border_mode={self.border_mode}, ival={self.ival}, ' \
           f'mval={self.mval}, always_apply={self.always_apply}, p={self.p})'


class Pad(DualTransform):
    """Pad the input.

        Internally, the ``numpy.pad`` function is used. The ``border_mode``, ``ival`` and ``mval``
        arguments are forwarded to it. More details at:
        https://numpy.org/doc/stable/reference/generated/numpy.pad.html.

        Args:
            pad_size (int | Tuple[int]): Number of pixels padded to the edges of each axis.

                Must be either of: ``P``, ``(P1, P2)``, or ``(P_Z1, P_Z2, P_Y1, P_Y2, P_X1, P_X2)``.

                If an integer, it is equivalent to ``(P, P, P, P, P, P)``.

                If a tuple of two numbers, it is equivalent to ``(P1, P2, P1, P2, P1, P2)``.

                Otherwise, it must specify padding for all spatial dimensions.

                The unspecified dimensions (C and T) are not affected.
            border_mode (str, optional): Values outside image domain are filled according to this mode.

                Defaults to ``'constant'``.
            ival (float | Sequence, optional): Values of `image` voxels outside of the `image` domain.
                Only applied when ``border_mode = 'constant'`` or ``border_mode = 'linear_ramp'``.

                Defaults to ``0``.
            mval (float | Sequence, optional): Values of `mask` voxels outside of the `mask` domain.
                Only applied when ``border_mode = 'constant'`` or ``border_mode = 'linear_ramp'``.

                Defaults to ``0``.
            ignore_index (float | None, optional): If specified, then transformation of `mask` is done with
                ``border_mode = 'constant'`` and ``mval = ignore_index``.

                If ``None``, this argument is ignored.

                Defaults to ``None``.
            always_apply (bool, optional): Always apply this transformation in composition.

                Defaults to ``True``.
            p (float, optional): Probability of applying this transformation in composition.

                Defaults to ``1``.

        Targets:
            image, mask, float mask, key points, bounding boxes
    """

    def __init__(self, pad_size: Union[int, TypePairInt, TypeSextetInt],
                 border_mode: str = 'constant', ival: Union[float, Sequence] = 0, mval: Union[float, Sequence] = 0,
                 ignore_index: Union[float, None] = None, always_apply: bool = True, p: float = 1):
        super().__init__(always_apply, p)
        self.pad_size: TypeSextetInt = parse_pads(pad_size)
        self.border_mode = border_mode
        self.mask_mode = border_mode
        self.ival = ival
        self.mval = mval

        if not (ignore_index is None):
            self.mask_mode = 'constant'
            self.mval = ignore_index

    def apply(self, img, **params):
        return F.pad_pixels(img, self.pad_size, self.border_mode, self.ival)

    def apply_to_mask(self, mask, **params):
        return F.pad_pixels(mask, self.pad_size, self.mask_mode, self.mval, True)

    def apply_to_keypoints(self, keypoints, **params):
        return F.pad_keypoints(keypoints, self.pad_size)

    def __repr__(self):
        return f'Pad(pad_size={self.pad_size}, border_mode={self.border_mode}, ival={self.ival}, mval={self.mval}, ' \
               f'always_apply={self.always_apply}, p={self.p})'


##########################################################################################
#                                                                                        #
#                      INTENSITY-BASED TRANSFORMATIONS (LOCAL)                           #
#                                                                                        #
##########################################################################################

class GaussianBlur(ImageOnlyTransform):
    """Perform Gaussian blurring of an image.
        In case of a multi-channel image, individual channels are blured separately.

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
            p (float, optional): Probability of applying this transformation in composition.
            
                Defaults to ``0.5``.

        Targets:
            image
    """
    def __init__(self, sigma: Union[float, tuple, List[Union[tuple, float]]] = 0.8,
                 border_mode: str = 'reflect', cval: float = 0,
                 always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.sigma = sigma
        self.border_mode = border_mode
        self.cval = cval

    def apply(self, img, **params):
        return F.gaussian_blur(img, self.sigma, self.border_mode, self.cval)

    def __repr__(self):
        return f'GaussianBlur(sigma={self.sigma}, border_mode={self.border_mode}, cval={self.cval}, ' \
               f'always_apply={self.always_apply}, p={self.p})'


class RandomGaussianBlur(ImageOnlyTransform):
    """Perform Gaussian blurring of an image with a random blurring strength.
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
            p (float, optional): Probability of applying this transformation in composition.
            
                Defaults to ``0.5``.

        Targets:
            image
    """
    def __init__(self, max_sigma: Union[float, tuple, List[Union[float, tuple]]] = 0.8,
                 min_sigma: float = 0, border_mode: str = 'reflect', cval: float = 0,
                 always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.max_sigma = max_sigma  # parse_coefs(max_sigma, d4=True)
        self.min_sigma = min_sigma
        self.border_mode = border_mode
        self.cval = cval

    def apply(self, img, **params):
        return F.gaussian_blur(img, params['sigma'], self.border_mode, self.cval)

    def get_params(self, targets, **data):
        if isinstance(self.max_sigma, (float, int, tuple)):
            # Randomly choose a single sigma for all axes and channels OR a sigma for each axis (except the C axis)
            sigma = get_sigma_axiswise(self.min_sigma, self.max_sigma)
        else:
            # max_sigma is list --> randomly choose sigmas for each channel
            sigma = [get_sigma_axiswise(self.min_sigma, channel) for channel in self.max_sigma]
        return {'sigma': sigma}

    def __repr__(self):
        return f'RandomGaussianBlur(max_sigma={self.max_sigma}, min_sigma={self.min_sigma}, ' \
               f'border_mode={self.border_mode}, cval={self.cval}, always_apply={self.always_apply}, p={self.p})'


class RemoveBackgroundGaussian(ImageOnlyTransform):
    """
    Remove background from the input image by subtracting a blurred image from the original image.

    The background image is created using Gaussian blurring. In case of a multi-channel image, individual channels
    are blured separately.

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

            Defaults to ``10``.
        mode (str, optional): Mode of background removal. Possible values:
            ``'default'`` (subtract blurred image from the input image),
            ``'bright_objects'`` (subtract the point-wise minimum of (blurred image, input image) from the input image),
            ``'dark_objects'`` (subtract the input image from the point-wise maximum of (blurred image, input image)).

            Defaults to ``'default'``.
        border_mode (str, optional): Values outside image domain are filled according to this mode.

            Defaults to ``'reflect'``.
        cval (float, optional): Value to fill past edges of image. Only applied when ``border_mode = 'constant'``.

            Defaults to ``0``.
        always_apply (bool, optional): Always apply this transformation in composition.

            Defaults to ``True``.
        p (float, optional): Probability of applying this transformation in composition.

            Defaults to ``1.0``.

    Targets:
        image
    """

    def __init__(self, sigma: Union[float, tuple, List[Union[tuple, float]]] = 10, mode: str = 'default',
                 border_mode: str = 'reflect', cval: float = 0,
                 always_apply: bool = True, p: float = 1.0):

        super().__init__(always_apply, p)
        self.sigma = sigma
        self.mode = mode
        self.border_mode = border_mode
        self.cval = cval

    def apply(self, img, **params):
        background = F.gaussian_blur(img, self.sigma, self.border_mode, self.cval)

        if self.mode == 'bright_objects':
            return img - np.minimum(background, img)

        if self.mode == 'dark_objects':
            return np.maximum(background, img) - img

        return img - background

    def __repr__(self):
        return f'RemoveBackgroundGaussian(sigma={self.sigma}, mode={self.mode}, border_mode={self.border_mode}, ' \
               f'cval={self.cval}, always_apply={self.always_apply}, p={self.p})'


##########################################################################################
#                                                                                        #
#                      INTENSITY-BASED TRANSFORMATIONS (POINT)                           #
#                                                                                        #
##########################################################################################

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
            p (float, optional): Probability of applying this transformation in composition.
            
                Defaults to ``0.5``.

        Targets:
            image
    """
    def __init__(self, brightness_limit: Union[float, TypePairFloat] = 0.2,
                 contrast_limit: Union[float, TypePairFloat] = 0.2,
                 always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.brightness_limit = to_tuple(brightness_limit)
        self.contrast_limit = to_tuple(contrast_limit)

    def apply(self, img, **params):
        return F.brightness_contrast_adjust(img, params['alpha'], params['beta'])

    def get_params(self, targets, **data):
        # Get transformation parameters:
        return {
            'alpha': 1.0 + uniform(self.contrast_limit[0], self.contrast_limit[1]),
            'beta': 0.0 + uniform(self.brightness_limit[0], self.brightness_limit[1]),
        }

    def __repr__(self):
        return f'RandomBrightnessContrast(brightness_limit={self.brightness_limit}, ' \
               f'contrast_limit={self.contrast_limit}, always_apply={self.always_apply}, p={self.p})'


class RandomGamma(ImageOnlyTransform):
    """Perform the gamma transformation with a randomly chosen gamma.

        Note:
            If image values (in any channel) are outside the [0,1] interval, this transformation is not performed.

        Args:
            gamma_limit (Tuple(float), optional): Interval from which gamma is selected.

                Defaults to ``(0.8, 1.2)``.
            always_apply (bool, optional): Always apply this transformation in composition.

                Defaults to ``False``.
            p (float, optional): Probability of applying this transformation in composition.

                Defaults to ``0.5``.

        Targets:
            image
    """

    def __init__(self, gamma_limit: TypePairFloat = (0.8, 1.2),
                 always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.gamma_limit = gamma_limit

    def apply(self, img, gamma=1, **params):
        return F.gamma_transform(img, gamma=gamma)

    def get_params(self, targets, **data):
        return {'gamma': uniform(self.gamma_limit[0], self.gamma_limit[1])}

    def __repr__(self):
        return f'RandomGamma(gamma_limit={self.gamma_limit}, always_apply={self.always_apply}, p={self.p})'


class HistogramEqualization(ImageOnlyTransform):
    """Equalize image histogram. The equalization is done channel-wise, meaning that each channel is equalized
        separately.

        Note:
            **Warning! Images are normalized over the spatial and temporal axes together. The output is in range [0, 1].**

        Args:
            bins (int, optional): Number of bins for image histogram.

                Defaults to ``256``.
            always_apply (bool, optional): Always apply this transformation in composition. 
            
                Defaults to ``False``.
            p (float, optional): Probability of applying this transformation in composition.
            
                Defaults to ``1``.

        Targets:
            image
    """
    def __init__(self, bins: int = 256, always_apply: bool = False, p: float = 1):
        super().__init__(always_apply, p)
        self.bins = bins

    def apply(self, img, **params):
        return F.histogram_equalization(img, self.bins)

    def __repr__(self):
        return f'HistogramEqualization(bins={self.bins}, always_apply={self.always_apply}, p={self.p})'


# TODO potential upgrade : different sigmas for different channels
class GaussianNoise(ImageOnlyTransform):
    """Add Gaussian noise to the image. The noise is drawn from normal distribution with given parameters.

        Args:
            var_limit (tuple, optional): Variance of normal distribution is randomly chosen from this interval.

                Defaults to ``(0.001, 0.1)``.
            mean (float, optional): Mean of normal distribution.

                Defaults to ``0``.
            always_apply (bool, optional): Always apply this transformation in composition.

                Defaults to ``False``.
            p (float, optional): Probability of applying this transformation in composition.

                Defaults to ``0.5``.

        Targets:
            image
    """

    def __init__(self, var_limit: TypePairFloat = (0.001, 0.1), mean: float = 0,
                 always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.var_limit = var_limit
        self.mean = mean

    def apply(self, img, **params):
        return F.gaussian_noise(img, sigma=params['sigma'], mean=self.mean)

    def get_params(self, targets, **params):
        # Choose noise standard deviation randomly (noise mean is given deterministically)
        var = uniform(self.var_limit[0], self.var_limit[1])
        sigma = var ** 0.5
        return {'sigma': sigma}

    def __repr__(self):
        return f'GaussianNoise(var_limit={self.var_limit}, mean={self.mean}, ' \
               f'always_apply={self.always_apply}, p={self.p})'


class PoissonNoise(ImageOnlyTransform):
    """Add Poisson noise to the image.

        Args:
            peak_limit (tuple): Range to sample the expected intensity of Poisson noise.

                Defaults to ``(0.1, 0.5)``.
            always_apply (bool, optional): Always apply this transformation in composition.

                Defaults to ``False``.
            p (float, optional): Probability of applying this transformation in composition.

                Defaults to ``0.5``.

        Targets:
            image
    """

    def __init__(self, peak_limit: TypePairFloat = (0.1, 0.5),
                 always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.peak_limit = peak_limit

    def apply(self, img, **params):
        return F.poisson_noise(img, peak=params['peak'])

    def get_params(self, targets, **params):
        peak = uniform(self.peak_limit[0], self.peak_limit[1])
        return {'peak': peak}

    def __repr__(self):
        return f'PoissonNoise(peak_limit={self.peak_limit}, always_apply={self.always_apply}, p={self.p})'


class Normalize(ImageOnlyTransform):
    """Change image mean and standard deviation to the given values (channel-wise).

        Args:
            mean (float | List[float], optional): The desired channel-wise means.

                Must be either of: ``M`` (for single-channel images),
                ``[M_1, M_2, ..., M_C]`` (for multi-channel images).

                Defaults to ``0``.
            std (float | List[float], optional): The desired channel-wise standard deviations.

                Must be either of: ``S`` (for single-channel images),
                ``[S_1, S_2, ..., S_C]`` (for multi-channel images).

                Defaults to ``1``.
            always_apply (bool, optional): Always apply this transformation in composition. 
            
                Defaults to ``True``.
            p (float, optional): Probability of applying this transformation in composition.
            
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
        return f'Normalize(mean={self.mean}, std={self.std}, always_apply={self.always_apply}, p={self.p})'


# TODO create checks (mean, std, got good shape, and etc.), what if given list but only one channel, and reverse.
class NormalizeMeanStd(ImageOnlyTransform):
    """Normalize image values to mean 0 and standard deviation 1, given the channel-wise means and standard deviations.

        For a single-channel image, the normalization is applied by the formula: :math:`img = (img - mean) / std`.
        If the image contains more channels, then the formula is used for each channel separately.

        It is recommended to input dataset-wide means and standard deviations.

        Args:
            mean (float | List[float]): Channel-wise image mean.

                Must be either of: ``M`` (for single-channel images),
                ``(M_1, M_2, ..., M_C)`` (for multi-channel images).
            std (float | List[float]): Channel-wise image standard deviation.

                Must be either of: ``S`` (for single-channel images),
                ``(S_1, S_2, ..., S_C)`` (for multi-channel images).
            always_apply (bool, optional): Always apply this transformation in composition.

                Defaults to ``True``.
            p (float, optional): Probability of applying this transformation in composition.

                Defaults to ``1``.

        Targets:
            image
    """

    def __init__(self, mean: Union[tuple, float], std: Union[tuple, float],
                 always_apply: bool = True, p: float = 1.0):
        super().__init__(always_apply, p)
        self.mean: np.ndarray = np.array(mean, dtype=np.float32)
        self.std: np.ndarray = np.array(std, dtype=np.float32)
        assert self.mean.shape == self.std.shape
        # Compute the formula denominator once as it is computationally expensive:
        self.denominator = np.reciprocal(self.std, dtype=np.float32)

        if len(self.mean.shape) == 0:  # shapes of self.mean and self.denominator are the same
            self.mean = self.mean[..., None]
            self.denominator = self.denominator[..., None]

    def apply(self, image, **params):
        return F.normalize_mean_std(image, self.mean, self.denominator)

    def __repr__(self):
        return f'NormalizeMeanStd(mean={self.mean}, std={self.std}, always_apply={self.always_apply}, p={self.p})'


##########################################################################################
#                                                                                        #
#                                   OTHER TRANSFORMATIONS                                #
#                                                                                        #
##########################################################################################

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
        return f'Float(always_apply={self.always_apply}, p={self.p})'



