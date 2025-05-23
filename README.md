# Bio-Volumentations

`Bio-Volumentations` is an **image augmentation and preprocessing package** for 3D (volumetric), 
4D (time-lapse volumetric or multi-channel volumetric), and 5D (time-lapse multi-channel volumetric) 
biomedical images and their annotations.

The library offers a wide range of efficiently implemented image transformations.
This includes both preprocessing transformations (such as intensity normalisation and padding) 
and augmentation transformations (such as affine transform, noise addition and removal, and contrast manipulation).


# Why use Bio-Volumentations?

`Bio-Volumentations` are a handy tool for image manipulation in machine learning applications. 
The library can transform **3D to 5D images** with **image-based and point-based annotations**, 
gives you **fine-grained control** over the transformation pipelines, 
and can be used with **any major Python deep learning library** 
(including PyTorch, PyTorch Lightning, TensorFlow, and Keras) 
in **a wide range of applications** including classification, object detection, semantic & instance 
segmentation, and object tracking.

`Bio-Volumentations` build upon widely used libraries such as Albumentations and TorchIO 
(see the _Contributions and Acknowledgements_ section below) and are accompanied by 
[detailed documentation and a user guide](https://biovolumentations.readthedocs.io/1.3.2/). 
Therefore, they can easily be adopted by developers.


# Installation

Simply install the package from pip using:
```commandline
pip install bio-volumentations
```

That's it :)

For more details, see [the project's PyPI page](https://pypi.org/project/bio-volumentations/).

Releases can also be found at [the project's GitHub page](https://github.com/xluciadh/bio-volumentations/releases).

### Requirements

- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Scikit-image](https://scikit-image.org/)
- [SimpleITK](https://simpleitk.org/)


### GitLab or GitHub?

`Bio-Volumentations` are being actively developed by the [CBIA @ FI MUNI](https://cbia.fi.muni.cz/), therefore
CBIA's official repository at [GitLab](https://gitlab.fi.muni.cz/cbia/bio-volumentations) is the primary repository
at the moment. The project's [GitHub repository](https://github.com/xluciadh/bio-volumentations) holds releases 
of our library and will be updated at important milestones and with new library versions.

# Usage

### The First Example

To check out our library on test data, you can run the example provided in the `example` folder.

There, you will find a test sample consisting of a 3D image (`image.tif`) with an associated binary mask
(`segmentation_mask.tif`), a runnable Python script, and the transformed sample (`image_transformed.tif` and 
`segmentation_mask_transformed.tif`).

To run the example, please download the `example` folder and 
install the `bio-volumentations`, `tiffile` and `imagecodecs` packages to your Python environment. 
Then run the following from the command line:

```commandline
cd example
python transformation_example.py
```

The script will generate a new randomly transformed sample and save it into the `image_transformed.tif` and 
`segmentation_mask_transformed.tif` files. These files can be opened using ImageJ.

This example uses data from the _Fluo-N3DH-CE_ dataset [1] from the Cell Tracking Challenge repository [2].

[1] Murray J, Bao Z, Boyle T, et al. Automated analysis of embryonic gene expression with cellular 
resolution in _C. elegans_. _Nat Methods_ 2008;**5**:703–709. https://doi.org/10.1038/nmeth.1228.

[2] Maška M, Ulman V, Delgado-Rodriguez P, et al. The Cell Tracking Challenge: 10 years of objective 
benchmarking. _Nat Methods_ 2023;**20**:1010–1020. https://doi.org/10.1038/s41592-023-01879-y.
Repository: https://celltrackingchallenge.net/3d-datasets/.

### Importing

Import the library to your project using:
```python
import bio_volumentations as biovol
```

### How to Use Bio-Volumentations?

The `Bio-Volumentations` library processes 3D, 4D, and 5D images. Each image must be 
represented as a `numpy.ndarray` and must conform to the following conventions:

- The order of dimensions is [C, Z, Y, X, T], where C is the channel dimension, 
   T is the time dimension, and Z, Y, and X are the spatial dimensions.
- The three spatial dimensions (Z, Y, X) must be present. To transform a 2D image, please create a dummy Z dimension first. 
- The channel (C) dimension is optional. If it is not present, the library will automatically
   create a dummy dimension in its place, so the output image shape will be [1, Z, Y, X].
- The time (T) dimension is optional and can only be present if the channel (C) dimension is 
   also present in the input data. To process single-channel time-lapse images, please create a dummy C dimension.

Thus, an input image is interpreted in the following ways based on its dimensionality:

1. 3D: a single-channel volumetric image [Z, Y, X];
2. 4D: a multi-channel volumetric image [C, Z, Y, X];
3. 5D: a single- or multi-channel volumetric image sequence [C, Z, Y, X, T].

The shape of the output image is either [C, Z, Y, X] (cases 1 & 2) or [C, Z, Y, X, T] (case 3).

The images are type-casted to a floating-point datatype before being transformed, irrespective of their actual datatype.

For the specification of image annotation conventions, please see below.

The transformations are implemented as callable classes inheriting from an abstract `Transform` class.
Upon instantiating a transformation object, one has to specify the parameters of the transformation.

All transformations work in a fully 3D fashion. Individual channels and time points of a data volume
are usually transformed separately and in the same manner; however, certain transformations can also work
along these dimensions. For instance, `GaussianBlur` can perform the blurring along the temporal dimension and
with different strength in individual channels.

The data can be transformed by a call to the transformation object.
**It is strongly recommended to use `Compose` to create and use transformation pipelines.** <br>
An instantiated `Compose` object encapsulates the full transformation pipeline and provides additional support:
it automatically checks and adjusts image format and datatype, outputs the image as a contiguous array, and
can optionally convert the transformed image to a desired format.
If you call transformations outside of `Compose`, we cannot guarantee the all assumptions
are checked and enforced, so you might encounter unexpected behaviour.

Below, there are several examples of how to use this library. You are also welcome to check 
[our documentation pages](https://biovolumentations.readthedocs.io/1.3.2/).

### Example: Transforming a Single Image

To create the transformation pipeline, you just need to instantiate all desired transformations
(with the desired parameter values)
and then feed a list of these transformation objects into a new `Compose` object. 

Optionally, you can specify a datatype conversion transformation that will be applied after the last transformation
in the list, e.g. from the default `numpy.ndarray` to a `torch.Tensor`. You can also specify the probability
of actually applying the whole pipeline as a number between 0 and 1. 
The default probability is 1 (i.e., the pipeline is applied in each call).
See the [docs](https://biovolumentations.readthedocs.io/1.3.2/examples.html) for more details.

The `Compose` object is callable. The data is passed as a keyword argument, and the call returns a dictionary 
with the same keyword and the corresponding transformed image. This might look like an overkill for a single image, 
but it will come handy when transforming images with annotations. The default key for an image is `'image'`.


```python
import numpy as np
from bio_volumentations import Compose, RandomGamma, RandomRotate90, GaussianBlur

# Create the transformation pipeline using Compose
aug = Compose([
        RandomGamma(gamma_limit = (0.8, 1.2), p = 0.8),
        RandomRotate90(axes = [1, 2, 3], p = 1),
        GaussianBlur(sigma = 1.2, p = 0.8)
      ])

# Generate an image - shape [C, Z, Y, X]
img = np.random.rand(1, 128, 256, 256)

# Transform the image
# Please note that the image must be passed as a keyword argument to the transformation pipeline
# and extracted from the outputted dictionary.
data = {'image': img}
aug_data = aug(**data)
transformed_img = aug_data['image']
```

### Example: Transforming Images with Annotations

Sometimes, it is necessary to transform an image with some associated additional targets.
To that end, `Bio-Volumentations` define several target types:

- `image` for the image data;
- `mask` for integer-valued label images;
- `float_mask` for real-valued label images;
- `keypoints` for a list of key points; and
- `value` for non-transformed values.

For more information on the format of individual target types, see the 
[Getting Started guide](https://biovolumentations.readthedocs.io/1.3.2/examples.html#example-transforming-images-with-annotations)

Please note that there must always be an `image`-type target in the sample.

If a `Random...` transform receives multiple targets on its input in a single call,
the same transformation parameters are used to transform all of these targets.
For example, `RandomAffineTransform` applies the same geometric transformation to all target types in a single call.

Some transformations, such as `RandomGaussianNoise` or `RandomGamma`, are only defined for the `image` target 
and leave the other targets unchanged. Please consult the 
[documentation of the individual transforms](https://biovolumentations.readthedocs.io/1.3.2/modules.html) for more details.

The associated targets are fed to the `Compose` object call as keyword arguments and extracted from the outputted
dictionary using the same keywords. 
The default key values are `'image'`, `'mask'`, `'float_mask'`, `'keypoints'`, and `'value'`.

```python
import numpy as np
from bio_volumentations import Compose, RandomGamma, RandomRotate90, GaussianBlur

# Create the transformation using Compose
aug = Compose([
        RandomGamma(gamma_limit = (0.8, 1.2), p = 0.8),
        RandomRotate90(axes = [1, 2, 3], p = 1),
        GaussianBlur(sigma = 1.2, p = 0.8)
      ])

# Generate image and a corresponding labeled image
img = np.random.rand(1, 128, 256, 256)
lbl = np.random.randint(0, 1, size=(128, 256, 256), dtype=np.uint8)

# Transform the images
# Please note that the images must be passed as keyword arguments to the transformation pipeline
# and extracted from the outputted dictionary.
data = {'image': img, 'mask': lbl}
aug_data = aug(**data)
transformed_img, transformed_lbl = aug_data['image'], aug_data['mask']
```


### Example: Transforming Multiple Targets of the Same Type

You can input arbitrary number of inputs to any transformation. To achieve this, you have to define the keywords
for the individual inputs when creating the `Compose` object.
The specified keywords will then be used to input the images to the transformation call as well as to extract the
transformed images from the outputted dictionary.

Specifically, you can define `image`-type target keywords using the `img_keywords` parameter - its value
must be a tuple of strings, each string representing a single keyword. Similarly, there are `mask_keywords`,
`fmask_keywords`, `value_keywords`, and `keypoints_keywords` parameters for the other target types. 
Setting any of these parameters overwrites its default value. 
The keywords can be any valid dictionary keys, and they must be unique.

You do not need to use all specified keywords in a transformation call. However, at least the target with
the `'image'` keyword must be present in each transformation call.
In our example below, we only transform three targets even though we defined four target keywords explicitly 
(and there are some implicit keywords as well for the other target types).

```python
import numpy as np
from bio_volumentations import Compose, RandomGamma, RandomRotate90, GaussianBlur

# Create the transformation using Compose: do not forget to define targets
aug = Compose([
        RandomGamma(gamma_limit = (0.8, 1.2), p = 0.8),
        RandomRotate90(axes = [1, 2, 3], p = 1),
        GaussianBlur(sigma = 1.2, p = 0.8)
    ],
    img_keywords=('image', 'abc'), mask_keywords=('mask',), fmask_keywords=('nothing',))

# Generate the image data: two images and a single int-valued mask
img = np.random.rand(1, 128, 256, 256)
img1 = np.random.rand(1, 128, 256, 256)
lbl = np.random.randint(0, 1, size=(128, 256, 256), dtype=np.uint8)

# Transform the images
# Please note that the images must be passed as keyword arguments to the transformation pipeline
# and extracted from the outputted dictionary.
data = {'image': img, 'abc': img1, 'mask': lbl}
aug_data = aug(**data)
transformed_img = aug_data['image']
transformed_img1 = aug_data['abc']
transformed_lbl = aug_data['mask']
```

### Example: Adding a Custom Transformation

Each transformation inherits from the `Transform` class. You can thus easily implement your own 
transformations and use them with this library. You can check our implementations to see how this can be done.
For example, `Flip` can be implemented as follows:

```python
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

    # Set transformation parameters. Useful especially for RandomXXX transforms to ensure consistent transformation of image tuples.
    def get_params(self, **data):
        axes = self.axes if self.axes is not None else [1, 2, 3]
        return {"axes": axes}
```

However, you cannot define your own target types, as that would require re-implementing all existing transforms.

# Implemented Transforms

### A List of Implemented Transformations

Intensity-based point transformations:
```python
Normalize
NormalizeMeanStd
HistogramEqualization 
GaussianNoise 
PoissonNoise
RandomBrightnessContrast 
RandomGamma
```

Intensity-based local transformations:
```python
GaussianBlur 
RandomGaussianBlur
RemoveBackgroundGaussian
```

Geometric (spatial) transformations:
```python
AffineTransform
Resize 
Scale
Rescale
Flip 
Pad
CenterCrop 
RandomAffineTransform
RandomScale 
RandomRotate90
RandomFlip 
RandomCrop
```

Other transformations:
```python
Contiguous
StandardizeDatatype
ConversionToFormat
```


### Runtime

Here, we present the execution times (mean ± standard deviation) of individual transformations from our library 
with respect to input image size.

The shape (size) of inputs was [1, 32, 32, 32, 1] (32k voxels), [4, 32, 32, 32, 5] (655k voxels), 
[4, 64, 64, 64, 5] (5M voxels), and [4, 128, 128, 128, 5] (42M voxels), respectively. 
The runtimes, presented in milliseconds, were averaged over 100 runs.
All measurements were done on a single workstation with a Ryzen 7-3700X CPU @ 3.60GHz.

| Transformation           | 32k voxels | 655k voxels |   5M voxels |    42M voxels |
|:-------------------------|-----------:|------------:|------------:|--------------:|
| AffineTransform          |   2 ± 0 ms |   23 ± 2 ms | 111 ± 11 ms |   988 ± 33 ms |
| RandomAffineTransform    |   3 ± 1 ms |   24 ± 1 ms |  110 ± 7 ms |  1005 ± 28 ms |
| Scale                    |   3 ± 1 ms |   25 ± 2 ms |  109 ± 9 ms |   990 ± 21 ms |
| RandomScale              |   3 ± 1 ms |   25 ± 2 ms |  111 ± 9 ms |  1000 ± 30 ms |
| Flip                     | < 1 ± 0 ms |    2 ± 0 ms |   14 ± 1 ms |    107 ± 8 ms |
| RandomFlip               | < 1 ± 0 ms |    1 ± 1 ms |   13 ± 5 ms |    90 ± 31 ms |
| RandomRotate90           | < 1 ± 0 ms |    2 ± 1 ms |   18 ± 5 ms |   150 ± 44 ms |
| GaussianBlur             |   1 ± 0 ms |   13 ± 1 ms |  150 ± 3 ms | 5501 ± 148 ms |
| RandomGaussianBlur       |   1 ± 0 ms |   11 ± 1 ms | 151 ± 18 ms | 5623 ± 176 ms |
| GaussianNoise            | < 1 ± 0 ms |   10 ± 1 ms |   83 ± 3 ms |    687 ± 8 ms |
| PoissonNoise             |   1 ± 0 ms |   20 ± 1 ms |  160 ± 1 ms |  1285 ± 18 ms |
| HistogramEqualization    |   2 ± 0 ms |   35 ± 1 ms |  266 ± 5 ms |  2120 ± 18 ms |
| Normalize                | < 1 ± 0 ms |    2 ± 1 ms |   32 ± 5 ms |   346 ± 24 ms |
| NormalizeMeanStd         | < 1 ± 0 ms |    1 ± 0 ms |    9 ± 1 ms |     69 ± 5 ms |
| RandomBrightnessContrast | < 1 ± 0 ms |    1 ± 0 ms |    6 ± 1 ms |     49 ± 8 ms |
| RandomGamma              | < 1 ± 0 ms |    7 ± 0 ms |   59 ± 1 ms |    469 ± 6 ms |


### Runtime: Comparison to Other Libraries

We also present the execution times (mean ± standard deviation) of eight commonly used transformations, 
comparing the performance of our `Bio-Volumentations` to other libraries capable of processing volumetric 
image data: `TorchIO` [3], `Volumentations` [4, 5], and `Gunpowder` [6].

Asterisks (*) denote transformations that only partially correspond to the desired functionality. 
Dashes (-) denote transformations that are missing from the respective library. 
The fastest implementation of each transformation is highlighted in bold.
The runtimes, presented in milliseconds, were averaged over 100 runs.
All measurements were done with a single-channel volumetric input image of size (256, 256, 256) 
on a single workstation with a Ryzen 7-3700X CPU @ 3.60GHz.

| Transformation                 |       `TorchIO` | `Volumentations` |   `Gunpowder` | `Bio-Volumentations` |
|:-------------------------------|----------------:|-----------------:|--------------:|---------------------:|
| Cropping                       |      *27 ± 3 ms |        20 ± 3 ms |  **6 ± 1 ms** |            21 ± 3 ms |
| Padding                        |      64 ± 10 ms |    **32 ± 6 ms** |     64 ± 4 ms |            43 ± 2 ms |
| Flipping                       |      50 ± 13 ms |       40 ± 10 ms |     36 ± 3 ms |       **31 ± 11 ms** |
| Affine transform               | **860 ± 24 ms** |   *3964 ± 816 ms |             - |         2595 ± 39 ms |
| Affine transform (anisotropic) |               - |                - |             - |     **2608 ± 36 ms** |
| Gaussian blur                  |   4417 ± 103 ms |                - |             - |     **3017 ± 84 ms** |
| Gaussian noise                 |  **183 ± 7 ms** |       393 ± 5 ms |  *425 ± 14 ms |           266 ± 8 ms |
| Brightness and contrast change |               - |        73 ± 2 ms |    197 ± 6 ms |        **29 ± 4 ms** |
| Z-normalization                |     219 ± 11 ms |  **124 ± 19 ms** |             - |          136 ± 13 ms |

[3] Pérez-García F, Sparks R, Ourselin S. TorchIO: A Python library for efficient loading, 
preprocessing, augmentation and patch-based sampling of medical images in deep learning. 
_Comput Meth Prog Bio_ 2021;**208**:106236. https://www.sciencedirect.com/science/article/pii/S0169260721003102

[4] Volumentations maintainers and contributors. Volumentations 3D. Version 1.0.4 [software]. 
GitHub, 2020 [cited 2024 Dec 16]. https://github.com/ZFTurbo/volumentations

[5] Solovyev R, Kalinin AA, Gabruseva T. 3D convolutional neural networks
for stalled brain capillary detection. _Comput Biol Med_ 2022;**141**:105089.
https://doi.org/10.1016/j.compbiomed.2021.105089

[6] Gunpowder maintainers and contributors. Gunpowder. Version 1.4.0 [software]. 
GitHub, 2024 [cited 2024 Dec 16]. https://github.com/funkelab/gunpowder

# Contributions and Acknowledgements

Authors of `Bio-Volumentations`: Samuel Šuľan, Lucia Hradecká, Filip Lux.
- Lucia Hradecká: lucia.d.hradecka@gmail.com   
- Filip Lux: lux.filip@gmail.com     

The `Bio-Volumentations` library is based on the following image augmentation libraries:
- [Albumentations](https://github.com/albumentations-team/albumentations)  
- [Volumentations](https://github.com/ashawkey/volumentations)                  
- [Volumentations: Continued Development](https://github.com/ZFTurbo/volumentations)                   
- [Volumentations: Enhancements](https://github.com/qubvel/volumentations)        
- [Volumentations: Further Enhancements](https://github.com/muellerdo/volumentations)
- [TorchIO](https://github.com/fepegar/torchio)

We would thus like to thank their authors, namely [the Albumentations team](https://github.com/albumentations-team), 
[Pavel Iakubovskii](https://github.com/qubvel), [ZFTurbo](https://github.com/ZFTurbo), 
[ashawkey](https://github.com/ashawkey), [Dominik Müller](https://github.com/muellerdo), and 
[TorchIO contributors](https://github.com/fepegar/torchio?tab=readme-ov-file#contributors).         


# Citation

If you find our library useful, please cite its [Zenodo record](https://doi.org/10.5281/zenodo.15023900) as:

```
Hradecká, L., & Lux, F. (2025). Bio-Volumentations (1.3.2). Zenodo. https://doi.org/10.5281/zenodo.15024087
```

and its [journal publication](https://doi.org/10.1016/j.softx.2025.102151) as:

```
Lucia Hradecká, Filip Lux, Samuel Šuľan, Petr Matula. Bio-Volumentations: A Python library for augmentation 
of volumetric image sequences. SoftwareX, 2025, vol. 30, 102151. DOI: 10.1016/j.softx.2025.102151.
```

Thank you! :)

