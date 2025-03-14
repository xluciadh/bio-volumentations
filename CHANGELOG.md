# Changelog

## 1.3.2

### Features

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
