Introduction
============
`Bio-Volumentations` is an image augmentation and preprocessing package for 3D (volumetric),
4D (time-lapse volumetric or multi-channel volumetric), and 5D (time-lapse multi-channel volumetric)
biomedical images and their annotations.

The library offers a wide range of efficiently implemented image transformations.
This includes both deterministic preprocessing transformations (such as intensity normalisation, padding, and type casting)
as well as random augmentation transformations (such as affine transform, noise addition and removal, and contrast manipulation).

The `Bio-Volumentations` library is a suitable tool for image data manipulation in machine learning applications.
It can transform several types of reference annotations along with the image data and
it can be used with any major Python deep learning library, including PyTorch, PyTorch Lightning, TensorFlow, and Keras.

This library builds upon wide-spread libraries such as Albumentations and TorchIO.
Therefore, it can easily be adopted by developers.

The source codes are available
`at the project's GitLab page <https://gitlab.fi.muni.cz/cbia/bio-volumentations/-/tree/1.3.0?ref_type=tags>`_.
The package can be also installed using pip - see `the project's PyPI page <https://pypi.org/project/bio-volumentations/>`_.

