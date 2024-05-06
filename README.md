# Bio-Volumentations

`Bio-Volumentations` is an image augmentation and preprocessing package for 3D (volumetric), 
4D (time-lapse volumetric or multi-channel volumetric), and 5D (time-lapse multi-channel volumetric) 
biomedical images and their annotations.

The library offers a wide range of efficiently implemented image transformations.
This includes both preprocessing transformations (such as intensity normalisation, padding, and type casting) 
and augmentation transformations (such as affine transform, noise addition and removal, and contrast manipulation).

The `Bio-Volumentations` library is a suitable tool for image manipulation in machine learning applications. 
It can be used with any major Python deep learning library, including PyTorch, PyTorch Lightning, TensorFlow, and Keras.

This library builds upon wide-spread libraries such as Albumentations and TorchIO (see the Contributions section below). 
Therefore, it can easily be adopted by developers.

# Installation

Install the package from pip using:
```python
pip install bio-volumentations
```

See [the project's PyPi page](https://pypi.org/project/bio-volumentations/) for more details.

## Requirements

- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Scikit-image](https://scikit-image.org/)
- [Matplotlib](https://matplotlib.org/)
- [SimpleITK](https://simpleitk.org/)


# Usage

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
   also present in the input data. To process single-channel time-lapse images, please create a dummy C dimension first.

Thus, an input image is interpreted in the following ways based on its shape:

1. [Z, Y, X] ... a single-channel volumetric image;
2. [C, Z, Y, X] ... a multi-channel volumetric image;
3. [C, Z, Y, X, T] ... a single-channel as well as multi-channel volumetric image sequence.

The shape of the output image is either [C, Z, Y, X] (cases 1 & 2) or [C, Z, Y, X, T] (case 3).

The images are type-casted to a floating-point datatype before transformations, irrespective of their actual datatype.

For the specification of image annotation conventions, please see below.

**It is strongly recommended to use `Compose` to create and use transformation pipelines.** <br>
The `Compose` class automatically checks and adjusts image format, datatype, stacks
individual transforms to a pipeline, and outputs the image as a contiguous array. 
Optionally, it can also convert the transformed image to a desired format. <br>
If you call transformations outside of `Compose`, we cannot guarantee the all assumptions are checked and enforced, 
so you might encounter unexpected behaviour.

Below, there are several examples of how to use this library. You are also welcome to check 
[our documentation pages](https://biovolumentations.readthedocs.io/1.2.0/).

### Example: Transforming a Single Image

To create the transformation pipeline, you just need to instantiate all desired transformations
(with the desired parameter values)
and then feed a list of these transformations into a new `Compose` object. 

Optionally, you can specify a datatype conversion transformation that will be applied after the last transformation
in the list, e.g. from the default `numpy.ndarray` to a `torch.Tensor`. You can also specify the probability
of actually applying the whole pipeline as a number between 0 and 1. The default probability is 1 (always apply).
See the [docs](https://biovolumentations.readthedocs.io/1.2.0/) for more details.

The `Compose` object is callable. The data is passed as a keyword argument, and the call returns a dictionary 
with the same keywords and corresponding transformed images. This might look like an overkill for a single image, 
but will come handy when transforming images with annotations. The default key for the image is `image`.


```python
import numpy as np
from bio_volumentations import Compose, RandomGamma, RandomRotate90, GaussianBlur

# Create the transformation pipeline using Compose
aug = Compose([
        RandomGamma(gamma_limit = (0.8, 1,2), p = 0.8),
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

### Example: Transforming Image Tuples

Sometimes, it is necessary to consistently transform a tuple of corresponding images.
To that end, `Bio-Volumentations` define several target types:

- `image` for the image data (any datatype allowed, gets converted to floating-point by default);
- `mask` for integer-valued label images (expected integer datatype); and
- `float_mask` for real-valued label images (expected floating-point datatype).

The `mask` and `float_mask` target types are expected to have the same shape as the `image`
target except for the channel (C) dimension which must not be included. 
For example, for images of shape [150, 300, 300], [1, 150, 300, 300], and
[4, 150, 300, 300], the corresponding `mask` and `float_mask` must be of shape [150, 300, 300].
If you want to use a multichannel `mask` or `float_mask`, you have to split it into
a set of single-channel `mask`s or `float_mask`s, respectively, and input them
as stand-alone targets (see below).

If a `Random...` transform receives multiple targets on its input in a single call,
the same transformation parameters are used to transform all of these targets.
For example, `RandomAffineTransform` applies the same geometric transformation to all target types in a single call.

Some transformations, such as `RandomGaussianNoise` or `RandomGamma`, are only defined for the `image` target 
and leave the `mask` and `float_mask` targets unchanged. Please consult the 
[documentation of the individual transforms](https://biovolumentations.readthedocs.io/1.2.0/) for more details.

The image tuples are fed to the `Compose` object call as keyword arguments and extracted from the outputted dictionary
using the same keys. The default key values are `image`, `mask`, and `float_mask`.

```python
import numpy as np
from bio_volumentations import Compose, RandomGamma, RandomRotate90, GaussianBlur

# Create the transformation using Compose
aug = Compose([
        RandomGamma(gamma_limit = (0.8, 1,2), p = 0.8),
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


### Example: Transforming Multiple Images of the Same Target Type

Although there are only three target types, you can input an arbitrary number of images to any
transformation. To achieve this, you have to define the value of the `targets` argument
when creating a `Compose` object.

The value of `targets` must be a list with exactly 3 items: a list with keys of `image`-type targets, 
a list with keys of `mask`-type targets, and 
a list with keys of `float_mask`-type targets. 
The specified keys will then be used to input the images to the transformation call as well as to extract the
transformed images from the outputted dictionary. 

The keys can be any valid dictionary keys; most importantly, they must be unique across all target types.
You don't need to feed an image for each target to the transformation call: in our example below, we have four targets
(two `image`, one `mask`, and one `float_mask`), but we only transform three images.

You cannot define your own target types; that would require re-implementing all existing transforms.

```python
import numpy as np
from bio_volumentations import Compose, RandomGamma, RandomRotate90, GaussianBlur

# Create the transformation using Compose: do not forget to define targets
aug = Compose([
        RandomGamma( gamma_limit = (0.8, 1,2), p = 0.8),
        RandomRotate90(axes = [1, 2, 3], p = 1),
        GaussianBlur(sigma = 1.2, p = 0.8)
    ], 
    targets= [ ['image' , 'image1'] , ['mask'], ['float_mask'] ])

# Generate the image data: two images and a single int-valued mask
img = np.random.rand(1, 128, 256, 256)
img1 = np.random.rand(1, 128, 256, 256)
lbl = np.random.randint(0, 1, size=(128, 256, 256), dtype=np.uint8)

# Transform the images
# Please note that the images must be passed as keyword arguments to the transformation pipeline
# and extracted from the outputted dictionary.
data = {'image': img, 'image1': img1, 'mask': lbl}
aug_data = aug(**data)
transformed_img = aug_data['image']
transformed_img1 = aug_data['image1']
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

    # Get transformation parameters. Useful especially for RandomXXX transforms to ensure consistent transformation of image tuples.
    def get_params(self, **data):
        axes = self.axes if self.axes is not None else [1, 2, 3]
        return {"axes": axes}
```


# Implemented Transforms

### A List of Implemented Transformations

Point transformations:
```python
GaussianNoise 
PoissonNoise
GaussianBlur 
RandomGaussianBlur
RandomGamma 
RandomBrightnessContrast 
HistogramEqualization 
Normalize
NormalizeMeanStd
```

Geometrical transformations:
```python
AffineTransform
Resize 
Scale
Flip 
CenterCrop 
Pad
RandomAffineTransform
RandomScale 
RandomRotate90
RandomFlip 
RandomCrop
```

Other:
```python
Float
Contiguous
```


# Contributions

Authors of the Bio-Volumentations library: Samuel Šuľan, Lucia Hradecká, Filip Lux.
- Lucia Hradecká: lucia.d.hradecka@gmail.com   
- Filip Lux: lux.filip@gmail.com     

The Bio-Volumentations library is based on the following image augmentation libraries:
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

TBA



