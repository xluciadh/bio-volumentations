Getting Started
===============

Installation
************
You can install the library from PyPi using ``pip install bio-volumentations``.

Importing
*********
Import the library to your project using ``import bio_volumentations as biovol``.

How to Use the Library?
***********************

The Bio-Volumentations library processes 3D, 4D, and 5D images. Each image must be
represented as :class:`numpy.ndarray` and must conform  to the following conventions:

- The order of dimensions is [C, Z, Y, X, T], where C is the channel dimension, T is the time dimension, and Z, Y, and X are the spatial dimensions.
- The three spatial dimensions (Z, Y, X) are compulsory.
- The channel (C) dimension is optional. If it is not present, the library will automatically create a dummy dimension in its place and output an image of shape (1, Z, Y, X).
- The time (T) dimension is optional and can only be present if the channel (C) dimension is also present.

Thus, the input images can have these shapes:

- [Z, Y, X] (a single-channel volumetric image)
- [C, Z, Y, X] (a multi-channel volumetric image)
- [C, Z, Y, X, T] (a single-channel as well as multi-channel volumetric image sequence)

**It is strongly recommended to use** :class:`Compose` **to create and use transformations.**
The :class:`Compose` class automatically checks and adjusts image format, datatype, stacks
individual transforms to a pipeline, and outputs the image as a contiguous array.
Optionally, it can also convert the transformed image to a desired format.

Below, there are several examples of how to use the `Bio-Volumentations` library.

Example: Transforming a Single Image
************************************
.. code-block:: python

    import numpy as np
    from bio_volumentations import Compose, RandomGamma, RandomRotate90, GaussianBlur

    # Create the transformation using Compose from a list of transformations
    aug = Compose([
            RandomGamma(gamma_limit = (0.8, 1,2), p = 0.8),
            RandomRotate90(axes = [1, 2, 3], p = 1),
            GaussianBlur(sigma = 1.2, p = 0.8)
          ])

    # Generate an image
    img = np.random.rand(1, 128, 256, 256)

    # Transform the image
    # Notice that the image must be passed as a keyword argument to the transformation pipeline
    # and extracted from the outputted dictionary.
    data = {'image': img}
    aug_data = aug(**data)
    transformed_img = aug_data['image']

Example: Transforming a Image Pairs
***********************************
Sometimes, it is necessary to consistently transform a tuple of corresponding images.
To that end, Bio-Volumentations define several target types:

- :class:`image` for the image data
- :class:`mask` for integer-valued label images
- :class:`float_mask` for real-valued label images

The :class:`mask` and :class:`float_mask` target types are expected to have the same shape as the :class:`image`
target except for the channel (C) dimension which must not be included.
For example, for images of shape (150, 300, 300), (1, 150, 300, 300), or
(4, 150, 300, 300), the corresponding :class:`mask` must be of shape (150, 300, 300).
If one wants to use a multichannel :class:`mask` or :class:`float_mask`, one has to split it into
a set of single-channel :class:`mask` s or :class:`float_mask` s, respectively, and input them
as stand-alone targets (see below).

If a :class:`Random...` transform receives multiple targets on its input in a single call,
the same random numbers are used to transform all of these targets.

However, some transformations might behave slightly differently for the individual
target types. For example, :class:`RandomCrop` works in the same way for all target types, while
:class:`RandomGaussianNoise` only affects the :class:`image` target and leaves the :class:`mask` and
:class:`float_mask` targets unchanged. Please consult the documentation of respective transforms
for more details.

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

The :class:`targets` must be a list with 3 items: a list with names of :class:`image`-type targets,
a list with names of :class:`mask`-type targets, and
a list with names of :class:`float_mask`-type targets. The specified names will then be used
to input the images to the transformation call as well as during extracting the
transformed images from the outputted dictionary. Please see the code below
for a practical example.

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

