import importlib.util
import sys
import os

bio_vol_package_path = '../volumentations_biomedicine/'  # the folder where folders "augmentations", "conversion", and "core" are located

spec = importlib.util.spec_from_file_location('volumentations-biomedicine',
                                              os.path.join(bio_vol_package_path,'__init__.py'))
BV = importlib.util.module_from_spec(spec)
sys.modules['volumentations-biomedicine'] = BV
spec.loader.exec_module(BV)

# check if it works
if __name__ == '__main__':
    print(BV.Compose)
