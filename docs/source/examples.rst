Getting Started
===============

Installation
************
You can install the `Bio-Volumentations` library from PyPi using:

``pip install bio-volumentations``

Required packages are:

- `NumPy <https://numpy.org/>`_
- `SciPy <https://scipy.org/>`_
- `Scikit-image <https://scikit-image.org/>`_
- `Matplotlib <https://matplotlib.org/>`_
- `SimpleITK <https://simpleitk.org/>`_

See `the project's PyPi page <https://pypi.org/project/bio-volumentations/>`_ for more details.

Importing
*********
You can import the `Bio-Volumentations` library to your project using:

.. code-block:: python

    import bio_volumentations as biovol

How to Use Bio-Volumentations?
******************************

The `Bio-Volumentations` library processes 3D, 4D, and 5D images. Each image must be
represented as :class:`numpy.ndarray` and must conform  to the following conventions:

- The order of dimensions is [C, Z, Y, X, T], where C is the channel dimension, T is the time dimension, and Z, Y, and X are the spatial dimensions.
- The three spatial dimensions (Z, Y, X) must be present. To transform a 2D image, please create a dummy Z dimension first.
- The channel (C) dimension is optional. If it is not present, the library will automatically create a dummy dimension in its place, so the output image shape will be [1, Z, Y, X].
- The time (T) dimension is optional and can only be present if the channel (C) dimension is also present in the input data. To process single-channel time-lapse images, please create a dummy C dimension first.

Thus, an input image is interpreted in the following ways based on its shape:
- [Z, Y, X] ... a single-channel volumetric image;
- [C, Z, Y, X] ... a multi-channel volumetric image;
- [C, Z, Y, X, T] ... a single-channel as well as multi-channel volumetric image sequence.

The shape of the output image is either [C, Z, Y, X] (cases 1 & 2) or [C, Z, Y, X, T] (case 3).

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

To create the transformation pipeline, you just need to instantiate all desired transformations
(with the desired parameter values)
and then feed a list of these transformations into a new :class:`Compose` object.

Optionally, you can specify a datatype conversion transformation that will be applied after the last transformation
in the list, e.g. from the default :class:`numpy.ndarray` to a :class:`torch.Tensor`. You can also specify the probability
of actually applying the whole pipeline as a number between 0 and 1. The default probability is 1 (always apply).
See the `docs <https://biovolumentations.readthedocs.io/1.2.0/>`_ for more details.

The :class:`Compose` object is callable. The data is passed as a keyword argument, and the call returns a dictionary
with the same keywords and corresponding transformed images. This might look like an overkill for a single image,
but will come handy when transforming images with annotations. The default key for the image is :class:`image`.

.. code-block:: python

    import numpy as np
    from bio_volumentations import Compose, RandomGamma, RandomRotate90, GaussianBlur

    # Create the transformation using Compose from a list of transformations
    aug = Compose([
            RandomGamma(gamma_limit = (0.8, 1,2), p = 0.8),
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
Sometimes, it is necessary to consistently transform a tuple of corresponding images.
To that end, `Bio-Volumentations` define several target types:

- :class:`image` for the image data (any datatype allowed, gets converted to floating-point by default);
- :class:`mask` for integer-valued label images (expected integer datatype); and
- :class:`float_mask` for real-valued label images (expected floating-point datatype).

The :class:`mask` and :class:`float_mask` target types are expected to have the same shape as the :class:`image`
target except for the channel (C) dimension which must not be included.
For example, for images of shape ``[150, 300, 300]``, ``[1, 150, 300, 300]``, and
``[4, 150, 300, 300]``, the corresponding :class:`mask` and :class:`float_mask` must be of shape ``[150, 300, 300]``.
If one wants to use a multi-channel :class:`mask` or :class:`float_mask`, one has to split it into
a set of single-channel :class:`mask` s or :class:`float_mask` s, respectively, and input them
as stand-alone targets (see below).

If a :class:`Random...` transform receives multiple targets on its input in a single call,
the same transformation parameters are used to transform all of these targets.
For example, :class:`RandomAffineTransform` applies the same geometric transformation to all target types in a single call.

Some transformations, such as :class:`RandomGaussianNoise` or :class:`RandomGamma`,
are only defined for the :class:`image` target
and leave the :class:`mask` and :class:`float_mask` targets unchanged. Please consult the
`documentation of the individual transforms <https://biovolumentations.readthedocs.io/1.2.0/>`_
for more details.

The image tuples are fed to the :class:`Compose` object call as keyword arguments and extracted from the outputted dictionary
using the same keys. The default key values are :class:`image`, :class:`mask`, and :class:`float_mask`.

.. code-block:: python

    import numpy as np
    from bio_volumentations import Compose, RandomGamma, RandomRotate90, GaussianBlur

    # Create the transformation using Compose from a list of transformations
    aug = Compose([
            RandomGamma(gamma_limit = (0.8, 1,2), p = 0.8),
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
Although there are only three target types, one input arbitrary number of images to any
transformation. To achieve this, one has to define the value of the :class:`targets` argument
when creating a :class:`Compose` object.

The value of :class:`targets` must be a list with exactly 3 items: a list with keys of :class:`image`-type targets,
a list with keys of :class:`mask`-type targets, and
a list with keys of :class:`float_mask`-type targets.
The specified keys will then be used to input the images to the transformation call as well as to extract the
transformed images from the outputted dictionary.

The keys can be any valid dictionary keys; most importantly, they must be unique across all target types.
You don't need to feed an image for each target to the transformation call: in our example below, we have four targets
(two :class:`image`, one :class:`mask`, and one :class:`float_mask`), but we only transform three images.

You cannot define your own target types; that would require re-implementing all existing transforms.


.. code-block:: python

    import numpy as np
    from bio_volumentations import Compose, RandomGamma, RandomRotate90, GaussianBlur

    # Create the transformation using Compose from a list of transformations and define targets
    aug = Compose([
            RandomGamma( gamma_limit = (0.8, 1,2), p = 0.8),
            RandomRotate90(axes = [1, 2, 3], p = 1),
            GaussianBlur(sigma = 1.2, p = 0.8)
        ],
        targets= [ ['image' , 'image1'] , ['mask'], ['float_mask'] ])

    # Generate the image data
    img = np.random.rand(1, 128, 256, 256)
    img1 = np.random.rand(1, 128, 256, 256)
    lbl = np.random.randint(0, 1, size=(128, 256, 256), dtype=np.uint8)

    # Transform the images
    # Notice that the images must be passed as keyword arguments to the transformation pipeline
    # and extracted from the outputted dictionary.
    data = {'image': img, 'image1': img1, 'mask': lbl}
    aug_data = aug(**data)
    transformed_img = aug_data['image']
    transformed_img1 = aug_data['image1']
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
        def __init__(self, axes: List[int] = None, always_apply=False, p=1):
            super().__init__(always_apply, p)
            self.axes = axes

        # Transform the image
        def apply(self, img, **params):
            return np.flip(img, params["axes"])

        # Transform the int-valued mask
        def apply_to_mask(self, mask, **params):
           # The mask has no channels
            return np.flip(mask, axis=[item - 1 for item in params["axes"]])

        # Transform the float-valued mask
        # By default, float_mask uses the implementation of mask, unless it is overridden (see the implementation of DualTransform).
        #def apply_to_float_mask(self, float_mask, **params):
        #    return self.apply_to_mask(float_mask, **params)

        # Get transformation parameters. Useful especially for RandomXXX transforms to ensure consistent transformation of image tuples.
        def get_params(self, **data):
            axes = self.axes if self.axes is not None else [1, 2, 3]
            return {"axes": axes}

