Getting Started
===============

Installation
************
You can install the `Bio-Volumentations` library from PyPI using:

``pip install bio-volumentations``

The required packages are:

- `NumPy <https://numpy.org/>`_
- `SciPy <https://scipy.org/>`_
- `Scikit-image <https://scikit-image.org/>`_
- `SimpleITK <https://simpleitk.org/>`_

See `the project's PyPI page <https://pypi.org/project/bio-volumentations/>`_ for more details.

Importing
*********
You can import the `Bio-Volumentations` library into your project using:

.. code-block:: python

    import bio_volumentations as biovol

How to Use Bio-Volumentations?
******************************

The `Bio-Volumentations` library processes 3D, 4D, and 5D images. Each image must be
represented as :class:`numpy.ndarray` and must conform to the following conventions:

- The order of dimensions is [C, Z, Y, X, T], where C is channel dimension, Z, Y, and X are spatial dimensions, and T is time dimension.
- The three spatial (Z, Y, X) dimensions must always be present. To transform a 2D image, please create a dummy Z dimension. (Or consider using a more suitable library, such as `Albumentations <https://albumentations.ai/>`_.)
- The channel (C) dimension is optional for the input image. However, the output image will always be at least 4-dimensional. If the C dimension is not present in the input, the library will automatically create a dummy dimension in its place, so the output image shape will be [1, Z, Y, X].
- The time (T) dimension is optional and can only be present if the channel (C) dimension is also present in the input data. To process single-channel time-lapse images, please create a dummy C dimension.

Thus, an input image is interpreted in the following ways based on its dimensionality:

- 3D: a single-channel volumetric image [Z, Y, X];
- 4D: a multi-channel volumetric image [C, Z, Y, X];
- 5D: a single- or multi-channel volumetric image sequence [C, Z, Y, X, T].

The shape of the output image will be either [C, Z, Y, X] (for cases 1 & 2) or [C, Z, Y, X, T] (for case 3).

The images are type-casted to a floating-point datatype before being transformed, irrespective of their actual datatype.

For the specification of image annotation conventions, please see
`the respective sections below <https://biovolumentations.readthedocs.io/1.2.0/examples.html#example-transforming-images-with-annotations>`_.

The transformations are implemented as callable classes inheriting from an abstract :class:`Transform` class.
Upon instantiating a transformation object, one has to specify the parameters of the transformation.

All transformations work in a fully 3D fashion. Individual channels and time points of a data volume
are usually transformed separately and in the same manner; however, certain transformations can also work
along these dimensions. For instance, :class:`GaussianBlur` can perform the blurring along the temporal dimension and
with different strength in individual channels.

The data can be transformed by a call to the transformation object.
**However, it is strongly recommended to use** :class:`Compose` **to create and use transformation pipelines.**
An instantiated :class:`Compose` object encapsulates the full transformation pipeline and provides additional support:
it automatically checks and adjusts image format and datatype, outputs the image as a contiguous array, and
can optionally convert the transformed image to a desired format.
If you call transformations outside of :class:`Compose`, we cannot guarantee the all assumptions
are checked and enforced, so you might encounter unexpected behaviour.

Below, there are several examples of how to use the `Bio-Volumentations` library. You are also welcome to check
`the API reference <https://biovolumentations.readthedocs.io/1.2.0/modules.html>`_ to learn more about the individual transforms.

Example: Transforming a Single Image
************************************

To create a transformation pipeline, you just need to instantiate all desired transformations
(with the desired parameter values)
and then feed a list of these transformations into a new :class:`Compose` object.

Optionally, you can specify a datatype conversion transformation that will be applied after the last transformation
in the list, for example from the default :class:`numpy.ndarray` to a PyTorch :class:`torch.Tensor`.
You can also specify the probability of applying the whole pipeline as a number between 0 and 1.
The default probability is 1; the pipeline is applied for each call. See the :class:`Compose`
`docs <https://biovolumentations.readthedocs.io/1.2.0/bio_volumentations.core.html#module-bio_volumentations.core.composition>`_
for more details.

Note: You can also toggle the probability of applying the individual transforms. To do so, you can
use the parameters :class:`p` and :class:`always_apply` when instantiating the transformation objects.
If :class:`always_apply==True`, the transformation is applied every time the pipeline is called;
otherwise, it is applied with probability :class:`p`, which must be a number between 0 and 1.

The :class:`Compose` object is callable. The data is passed as keyword arguments, and the call returns a dictionary
with the same keywords and corresponding transformed data. This might look like an overkill for a single image,
but it will come handy when transforming images with additional targets.
The default keyword for the image data is, unsurprisingly, :class:`'image'`.

.. code-block:: python

    import numpy as np
    from bio_volumentations import Compose, RandomGamma, RandomRotate90, GaussianBlur

    # Create the transformation using Compose from a list of transformations
    aug = Compose([
            RandomGamma(gamma_limit = (0.8, 1.2), p = 0.8),
            RandomRotate90(axes = [1, 2, 3], p = 1),
            GaussianBlur(sigma = 1.2, p = 0.8)
          ])

    # Generate an image - shape [C, Z, Y, X]
    img = np.random.rand(1, 128, 256, 256)

    # Transform the image
    # Notice that the image must be passed as a keyword argument to the transformation pipeline
    # and extracted from the outputted dictionary.
    data = {'image': img}
    aug_data = aug(**data)
    transformed_img = aug_data['image']

Example: Transforming Images with Annotations
*********************************************
Sometimes, it is necessary to transform an image with some corresponding additional targets.
To that end, `Bio-Volumentations` define several target types:

- :class:`image` for the image data (:class:`numpy.ndarray` with floating-point datatype);
- :class:`mask` for integer-valued label images (:class:`numpy.ndarray` with integer datatype);
- :class:`float_mask` for real-valued label images (:class:`numpy.ndarray` with floating-point datatype);
- :class:`keypoints` for a list of key points (a list of tuples); and
- :class:`value` for any non-transformed data.

Apart from these, a :class:`bounding_boxes` target type is defined but not implemented yet.

The :class:`mask` and :class:`float_mask` targets are expected to have the same shape as the :class:`image`
target except for the channel (C) dimension which must not be included.
For example, a :class:`mask` and/or :class:`float_mask` of shape ``[150, 300, 300]`` can correspond to
images of shape ``[150, 300, 300]``, ``[1, 150, 300, 300]``, as well as ``[4, 150, 300, 300]``.
If you want to use a multi-channel :class:`mask` or :class:`float_mask`, you have to split it into
a set of single-channel :class:`mask` s or :class:`float_mask` s, respectively, and input them
as stand-alone targets (see below how to transform multiple masks per image).

The :class:`keypoints` target is represented as a list of tuples. Each tuple represents
the absolute coordinates of a keypoint in the volume, so it must contain either 3 or 4 numbers
(for volumetric and time-lapse volumetric data, respectively).

The :class:`value` target can hold any other data whose value does not change during the transformation process.
This can be for example image-level information such as a classification labels.

If a :class:`Random...` transform receives multiple targets on its input in a single call,
the same transformation parameters are used to transform all of these targets.
For example, :class:`RandomAffineTransform` applies the same geometric transformation to all target types in a single call.

Some transformations, such as :class:`RandomGaussianNoise` or :class:`RandomGamma`,
are only defined for the :class:`image` target
and leave the other target types unchanged. Please consult the
`documentation of the individual transforms <https://biovolumentations.readthedocs.io/1.2.0/modules.html>`_
for more details.

The corresponding targets are fed to the :class:`Compose` object call as keyword arguments and extracted from the outputted
dictionary using the same keys. The default key values are :class:`'image'`, :class:`'mask'`, :class:`'float_mask'`,
:class:`'keypoints'`, :class:`'bboxes'`, and :class:`'value'`.

.. code-block:: python

    import numpy as np
    from bio_volumentations import Compose, RandomGamma, RandomRotate90, GaussianBlur

    # Create the transformation using Compose from a list of transformations
    aug = Compose([
            RandomGamma(gamma_limit = (0.8, 1.2), p = 0.8),
            RandomRotate90(axes = [1, 2, 3], p = 1),
            GaussianBlur(sigma = 1.2, p = 0.8)
          ])

    # Generate image and a corresponding labeled image
    img = np.random.rand(1, 128, 256, 256)
    lbl = np.random.randint(0, 1, size=(128, 256, 256), dtype=np.uint8)

    # Transform the images
    # Notice that the images must be passed as keyword arguments to the transformation pipeline
    # and extracted from the outputted dictionary.
    data = {'image': img, 'mask': lbl}
    aug_data = aug(**data)
    transformed_img, transformed_lbl = aug_data['image'], aug_data['mask']

Example: Transforming Multiple Images of the Same Target Type
*************************************************************
You can input arbitrary number of inputs to any transformation. To achieve this, you have to define the keywords
for the individual inputs when creating the :class:`Compose` object.
The specified keywords will then be used to input the images to the transformation call as well as to extract the
transformed images from the outputted dictionary.

Specifically, you can define :class:`image`-type target keywords using the :class:`img_keywords` parameter - its value
must be a tuple of strings, each string representing a single keyword. Similarly, there are :class:`mask_keywords`,
:class:`fmask_keywords`, :class:`keypoints_keywords`, :class:`bboxes_keywords`, and :class:`value_keywords` parameters
for the respective target types.

Importantly, there must always be an :class:`image`-type target with the keyword :class:`'image'`.
Otherwise, the keywords can be any valid dictionary keys, and they must be unique within each target type.

You do not need to use all specified keywords in a transformation call. However, at least the target with
the :class:`'image'` keyword must be present in each transformation call.
In our example below, there are seven target keywords defined: four keywords defined explicitly (two for :class:`image`,
one for :class:`mask`, and one for :class:`float_mask`) and three defined implicitly (for :class:`keypoints`,
:class:`bounding_boxes`, and :class:`value`), but we only transform three targets.

You cannot define your own target types; that would require re-implementing all existing transforms.


.. code-block:: python

    import numpy as np
    from bio_volumentations import Compose, RandomGamma, RandomRotate90, GaussianBlur

    # Create the transformation using Compose from a list of transformations and define targets
    aug = Compose([
            RandomGamma(gamma_limit = (0.8, 1.2), p = 0.8),
            RandomRotate90(axes = [1, 2, 3], p = 1),
            GaussianBlur(sigma = 1.2, p = 0.8)
        ],
        img_keywords=('image', 'abc'), mask_keywords=('mask',), fmask_keywords=('nothing',))

    # Generate the image data
    img = np.random.rand(1, 128, 256, 256)
    img1 = np.random.rand(1, 128, 256, 256)
    lbl = np.random.randint(0, 1, size=(128, 256, 256), dtype=np.uint8)

    # Transform the images
    # Notice that the images must be passed as keyword arguments to the transformation pipeline
    # and extracted from the outputted dictionary.
    data = {'image': img, 'abc': img1, 'mask': lbl}
    aug_data = aug(**data)
    transformed_img = aug_data['image']
    transformed_img1 = aug_data['abc']
    transformed_lbl = aug_data['mask']

Example: Adding a Custom Transformation
***************************************

Each transformation inherits from the :class:`Transform` class. You can thus easily implement your own
transformations and use them with this library. You can check our implementations to see how this can be done;
for example, :class:`Flip` can be implemented as follows:

.. code-block:: python

    import numpy as np
    from typing import List
    from bio_volumentations import DualTransform

    class Flip(DualTransform):
        # Initialize the transform
        def __init__(self, axes: List[int] = None, always_apply=False, p=1):
            super().__init__(always_apply, p)
            self.axes = axes

        # Transform the image
        def apply(self, img, **params):
            return np.flip(img, params["axes"])

        # Transform the int-valued mask
        def apply_to_mask(self, mask, **params):
            return np.flip(mask, axis=[item - 1 for item in params["axes"]])  # Mask has no channels

        # Transform the float-valued mask - no need to implement. By default, apply_to_float_mask() uses
        # the implementation of apply_to_mask(), unless it is overridden (see the implementation of DualTransform).

        # Transform the key points
        def apply_to_keypoints(self, keypoints, **params):
            return F.flip_keypoints(keypoints, axes=params['axes'], img_shape=params['img_shape'])

        # Get transformation parameters. This is useful especially for RandomXXX transforms
        # to ensure consistent transformation of samples with multiple targets.
        def get_params(self, **data):
            axes = [1, 2, 3] if self.axes is None else self.axes
            img_shape = np.array(data['image'].shape[1:4])
            return {"axes": axes, "img_shape": img_shape}

