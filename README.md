# Bio-Volumentations

`Bio-Volumentations` is an image augmentation and preprocessing package for 3D, 4D, and 5D biomedical images.

It offers a range of image transformations implemented efficiently for time-lapse multi-channel volumetric image data.
This includes both preprocessing transformations (such as intensity normalisation, padding, and type casting) 
and augmentation transformations (such as affine transform, noise addition and removal, and contrast manipulation).

The `Bio-Volumentations` library is a suitable tool for data manipulation in machine learning applications. 
It can be used with any major Python deep learning library, including PyTorch, PyTorch Lightning, TensorFlow, and Keras.

This library builds upon wide-spread libraries such as Albumentations (see the Contributions section below) 
in terms of design and user interface. Therefore, it can easily be adopted by developers.

# Installation

Install the package from pip using:
```python
pip install bio-volumentations
```
## Requirements

NumPy       https://numpy.org/ <br> 
SciPy       https://scipy.org/ <br>
Scikit-image https://scikit-image.org/ <br>
Matplotlib  https://matplotlib.org/ <br>
SimpleITK   https://simpleitk.org/ <br>


# Usage

### Importing

Import the library to your project using:
```python
import bio_volumentations as biovol
```

### How to Use Bio-Volumentations?

The Bio-Volumentations library processes 3D, 4D, and 5D images. Each image must be 
represented as `numpy.ndarray`s and must conform  to the following conventions:

- The order of dimensions is [C, Z, Y, X, T], where C is the channel dimension, 
   T is the time dimension, and Z, Y, and X are the spatial dimensions.
- The three spatial dimensions (Z, Y, X) are compulsory.
- The channel (C) dimension is optional. If it is not present, the library will automatically
   create a dummy dimension in its place and output an image of shape (1, Z, Y, X).
- The time (T) dimension is optional and can only be present if the channel (C) dimension is 
   also present.

Thus, the input images can have these shapes:

- [Z, Y, X] (a single-channel volumetric image)
- [C, Z, Y, X] (a multi-channel volumetric image)
- [C, Z, Y, X, T] (a single-channel as well as multi-channel volumetric image sequence)

**It is strongly recommended to use `Compose` to create and use transformation pipelines.** 
The `Compose` class automatically checks and adjusts image format, datatype, stacks
individual transforms to a pipeline, and outputs the image as a contiguous array. 
Optionally, it can also convert the transformed image to a desired format.

More at the [documentation pages](https://www.google.com).

Below, there are several examples of how to use this library.

### Example: Transforming a Single Image

```python
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
```

### Example: Transforming Image Tuples

Sometimes, it is necessary to consistently transform a tuple of corresponding images.
To that end, Bio-Volumentations define several target types:

- `image` for the image data;
- `mask` for integer-valued label images; and
- `float_mask` for real-valued label images.

The `mask` and `float_mask` target types are expected to have the same shape as the `image`
target except for the channel (C) dimension which must not be included. 
For example, for images of shape (150, 300, 300), (1, 150, 300, 300), or
(4, 150, 300, 300), the corresponding `mask` must be of shape (150, 300, 300).
If one wants to use a multichannel `mask` or `float_mask`, one has to split it into
a set of single-channel `mask`s or `float_mask`s, respectively, and input them
as stand-alone targets (see below).

If a `Random...` transform receives multiple targets on its input in a single call,
the same random numbers are used to transform all of these targets.

However, some transformations might behave slightly differently for the individual
target types. For example, `RandomCrop` works in the same way for all target types, while
`RandomGaussianNoise` only affects the `image` target and leaves the `mask` and
`float_mask` targets unchanged. Please consult the documentation of respective transforms
for more details.

```python
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
```


### Example: Transforming Multiple Images of the Same Target Type

Although there are only three target types, one input arbitrary number of images to any
transformation. To achieve this, one has to define the value of the `targets` argument
when creating a `Compose` object.

`targets` must be a list with 3 items: a list with names of `image`-type targets, 
a list with names of `mask`-type targets, and 
a list with names of `float_mask`-type targets. The specified names will then be used 
to input the images to the transformation call as well as during extracting the
transformed images from the outputted dictionary. Please see the code below 
for a practical example.

```python
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
- Albumentations:           https://github.com/albumentations-team/albumentations       
- 3D Conversion:            https://github.com/ashawkey/volumentations                  
- Continued Development:    https://github.com/ZFTurbo/volumentations                   
- Enhancements:             https://github.com/qubvel/volumentations                    
- Further Enhancements:     https://github.com/muellerdo/volumentations     

We would thus like to thank their authors, namely:
- The Albumentations team: https://github.com/albumentations-team                    
- Pavel Iakubovskii: https://github.com/qubvel                                 
- ZFTurbo: https://github.com/ZFTurbo                                
- ashawkey: https://github.com/ashawkey                               
- Dominik Müller: https://github.com/muellerdo         


# Citation

TBA



