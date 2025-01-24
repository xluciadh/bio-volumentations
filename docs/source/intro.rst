Introduction
============
`Bio-Volumentations` is an **image augmentation and preprocessing package** for 3D (volumetric),
4D (time-lapse volumetric or multi-channel volumetric), and 5D (time-lapse multi-channel volumetric)
biomedical images and their annotations.

The library offers **a wide range of efficiently implemented image transformations**
and fine-grained control over the transformation pipelines.
It comprises both deterministic preprocessing transformations (such as intensity normalisation and padding)
as well as random augmentation transformations (such as affine transformation, noise addition and removal, and contrast manipulation).

The `Bio-Volumentations` library is a suitable tool for image data manipulation in various machine learning applications,
including classification, object detection, semantic and instance segmentation, and object tracking.
It can transform **several types of reference annotations** along with the image data and
can be used with **any major Python deep learning library**, including PyTorch, PyTorch Lightning, TensorFlow, and Keras.

This library builds upon widely used libraries such as Albumentations and TorchIO, and can thus be easily adopted by developers.

The source codes are available
`at the project's GitLab page <https://gitlab.fi.muni.cz/cbia/bio-volumentations/-/tree/1.3.1?ref_type=tags>`_.
The package can be installed via `pip <https://pypi.org/project/bio-volumentations/>`_
and is archived `at GitHub <https://github.com/xluciadh/bio-volumentations/releases/tag/1.3.1>`_.

