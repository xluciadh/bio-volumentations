Getting Started
===============

Installation
************
You can install the `Bio-Volumentations` library from PyPI using:

``pip install bio-volumentations``

Required packages are:

- `NumPy <https://numpy.org/>`_
- `SciPy <https://scipy.org/>`_
- `Scikit-image <https://scikit-image.org/>`_
- `SimpleITK <https://simpleitk.org/>`_

See `the project's PyPI page <https://pypi.org/project/bio-volumentations/>`_ for more details.

Importing
*********
You can import the `Bio-Volumentations` library to your project using:

.. code-block:: python

    import bio_volumentations as biovol

How to Use Bio-Volumentations?
******************************

The `Bio-Volumentations` library processes 3D, 4D, and 5D images. Each image must be
represented as :class:`numpy.ndarray` and must conform to the following conventions:

- The order of dimensions is [C, Z, Y, X, T], where C is the channel dimension, T is the time dimension, and Z, Y, and X are the spatial dimensions.
- The three spatial dimensions (Z, Y, X) must be present. To transform a 2D image, please create a dummy Z dimension first.
- The channel (C) dimension is optional for the input image. However, the output image will always be at least 4-dimensional. If the C dimension is not present in the input, the library will automatically create a dummy dimension in its place, so the output image shape will be [1, Z, Y, X].
- The time (T) dimension is optional and can only be present if the channel (C) dimension is also present in the input data. To process single-channel time-lapse images, please create a dummy C dimension first.

Thus, an input image is interpreted in the following ways based on its dimensionality:

- 3D - a single-channel volumetric image [Z, Y, X];
- 4D - a multi-channel volumetric image [C, Z, Y, X];
- 5D - a multi-channel volumetric image sequence [C, Z, Y, X, T].

The shape of the output image will be either [C, Z, Y, X] (for cases 1 & 2) or [C, Z, Y, X, T] (for case 3).

The images are type-casted to a floating-point datatype before transformations, irrespective of their actual datatype.

For the specification of image annotation conventions, please see below.

**It is strongly recommended to use** :class:`Compose` **to create and use transformation pipelines.**
The :class:`Compose` class automatically checks and adjusts image format and datatype, stacks
individual transforms to a pipeline, and outputs the image as a contiguous array.
Optionally, it can also convert the transformed image to a desired format.
If you call transformations outside of :class:`Compose`, we cannot guarantee the all assumptions are checked and enforced,
so you might encounter unexpected behaviour.

Below, there are several examples of how to use the `Bio-Volumentations` library.
You are also welcome to check
`our documentation pages <https://biovolumentations.readthedocs.io/1.2.0/>`_.

Example: Transforming a Single Image
************************************

To create a transformation pipeline, you just need to instantiate all desired transformations
(with the desired parameter values)
and then feed a list of these transformations into a new :class:`Compose` object.

Optionally, you can specify a datatype conversion transformation that will be applied after the last transformation
in the list, for example from the default :class:`numpy.ndarray` to a PyTorch :class:`torch.Tensor`.
You can also specify the probability of actually applying the whole pipeline as a number between 0 and 1.
The default probability is 1 (always apply).
See the `docs <https://biovolumentations.readthedocs.io/1.2.0/>`_ for more details.

The :class:`Compose` object is callable. The data is passed as keyword arguments, and the call returns a dictionary
with the same keywords and corresponding transformed data. This might look like an overkill for a single image,
but it will come handy when transforming images with additional targets. The default key for the image is :class:`image`.

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

Example: Transforming Image Tuples
***********************************
Sometimes, it is necessary to consistently transform an image and its corresponding additional targets.
To that end, `Bio-Volumentations` define several target types:

- :class:`image` for the image data (:class:`numpy.ndarray` with any datatype allowed, gets converted to floating-point by default);
- :class:`mask` for integer-valued label images (:class:`numpy.ndarray` with integer datatype);
- :class:`float mask` for real-valued label images (:class:`numpy.ndarray` with floating-point datatype);
- :class:`value` for scalar values, such as classification labels (a floating-point number); and
- :class:`key points` for a list of key points (a list of tuples), where each key point is a tuple of 3 or 4 floating-point numbers (for volumetric and time-lapse volumetric data, respectively) that represent its absolute coordinates in the volume.

Apart from these, :class:`bounding boxes` target type is defined but not implemented yet.

The :class:`mask` and :class:`float mask` target types are expected to have the same shape as the :class:`image`
target except for the channel (C) dimension which must not be included.
For example, for images of shape ``[150, 300, 300]``, ``[1, 150, 300, 300]``, and
``[4, 150, 300, 300]``, the corresponding :class:`mask` and :class:`float mask` must be of shape ``[150, 300, 300]``.
If one wants to use a multi-channel :class:`mask` or :class:`float mask`, one has to split it into
a set of single-channel :class:`mask` s or :class:`float mask` s, respectively, and input them
as stand-alone targets (see below).

If a :class:`Random...` transform receives multiple targets on its input in a single call,
the same transformation parameters are used to transform all of these targets.
For example, :class:`RandomAffineTransform` applies the same geometric transformation to all target types in a single call.

Some transformations, such as :class:`RandomGaussianNoise` or :class:`RandomGamma`,
are only defined for the :class:`image` target
and leave the other target types unchanged. Please consult the
`documentation of the individual transforms <https://biovolumentations.readthedocs.io/1.2.0/>`_
for more details.

The image tuples are fed to the :class:`Compose` object call as keyword arguments and extracted from the outputted
dictionary using the same keys. The default key values are :class:`image`, :class:`mask`, :class:`float_mask`,
:class:`keypoints`, :class:`bboxes`, and :class:`class_value`.

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
You can input arbitrary number of inputs to any transformation. To achieve this, you has to define the keywords (names)
for the individual inputs when creating the :class:`Compose` object.

The value of :class:`targets` must be a list with exactly 3 items: a list with keys of :class:`image`-type targets,
a list with keys of :class:`mask`-type targets, and
a list with keys of :class:`float mask`-type targets.
The specified keys will then be used to input the images to the transformation call as well as to extract the
transformed images from the outputted dictionary.

Specifically, you can define :class:`image`-type target keywords using the :class:`img_keywords` parameter - its value
must be a tuple of strings, each string representing a single keyword. Similarly, there are the :class:`mask_keywords`,
:class:`fmask_keywords`, :class:`keypoints_keywords`, :class:`bboxes_keywords`, and :class:`value_keywords` parameters
for other target types.
Importantly, there must be an :class:`image`-type target with the keyword :class:`'image'`.
Otherwise, the keywords can be any valid dictionary keys, and they must be unique within each target type.

You do not need to use all specified keywords in a transformation call. However, at least the :class:`image`
keyword target must be present in each transformation call.
In our example below, there are seven target keywords defined: four keywords defined explicitly (two for :class:`image`,
one for :class:`mask`, and one for :class:`float mask`) and three defined implicitly (for :class:`value`,
:class:`key points`, and :class:`bounding boxes`), but we only transform three targets.

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
transformations and use them with this library. You can check our implementations to see how this can be done.
For example, :class:`Flip` can be implemented as follows:

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

