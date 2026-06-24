# Changelog

## 1.3.3 (Jun 23, 2026)

### New features

- Added bounding box support (`bboxes`-type targets) for all transformations
- Bounding boxes can be provided in the voc, coco, yolo, and albumentations formats
- `Compose` now ensures that keypoints are always outputted in the same format
  (a list of tuples with values of type int or float)
- The code is more robust: `Compose` performs more format checks (e.g. shape consistency checks, keypoint format fixes);
  more easy-to-understand warnings and errors are raised (e.g. target keywords validity, transformation parameter values 
  validity) to prevent code from failing or to fail as early as possible


### Changes

- Updated the example script to show how to work with keypoints and bounding boxes
- Refactored some parts of the code to increase its readability and efficiency
- Unified the default setting of `p` and `always_apply` for all transformations 
  (`always_apply=False` for all transforms except the "technical" ones, 
  `p` equals 1 for deterministic transformations and 0.5 for random transformations)
- Scaling transforms now only allow scaling by positive values
- Removed the unused napari_tools.py file
- Added the `keep_all` option for both keypoints and bounding boxes


### Fixes

- Fixed `RandomRotate90` to work correctly for all target types + you can now input multiple factors
- Fixed sitk backend to work correctly for all target types
- Time-lapse keypoints stopped losing their temporal position in specific transformations
- `CenterCrop` and `RandomCrop` now use image-specific `border_mode` and `ival` for image targets (different to masks)
- Minor fixes: pad_dims vs pad_size argument in `RandomCrop`, unused argument of `PoissonNoise`, etc.


## 1.3.2 (Mar 14, 2025)

### New features

- Lifted the requirement of the compulsory `'image'`-keyword target in each data sample.
  You can now use any valid keywords for your `image`-type targets 
- Added an example of using `Bio-Volumentations` with automatic augmentation strategies
  (`AutoAugment` and `RandAugment`)


### Changes

- Refactored the code
- Switched to using numpy random and vectorised operations to increase performance
- Improved code comments and warning messages
- Added more unit-tests (invalid input samples or parameters, keyword options, 
  vectorised implementation)
