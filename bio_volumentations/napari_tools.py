import numpy as np
import napari
from matplotlib import pyplot as plt
from skimage import exposure


def show_sample_np(img_np,
                   colormap=('green', 'red'),
                   scale=(1, 4, 1, 1)):

    viewer = napari.Viewer(ndisplay=3)
    
    n_dims = len(img_np.shape)
    if n_dims == 3:
        img_np = np.expand_dims(np.expand_dims(img_np, axis=0), axis=-1)
    elif (n_dims == 4) or (n_dims > 5) or (n_dims < 3):
        print(f'Warning: cannot decide about the image shape {img_np.shape}')
        return


    for c in range(img_np.shape[0]):


        im = img_np[c]

        pL, pH = np.percentile(img_np, (.5, 99.5))
        im = exposure.rescale_intensity(im, in_range=(pL, pH))


        im = np.moveaxis(im, -2, 0)
        im = np.moveaxis(im, -1, 0)

        viewer.add_image(im, name=f'ch{c}')
        viewer.layers[f'ch{c}'].interpolation = 'nearest'
        viewer.layers[f'ch{c}'].interpolation4d = 'nearest'
        viewer.layers[f'ch{c}'].interpolation2D = 'nearest'
        viewer.layers[f'ch{c}'].scale = scale
        viewer.layers[f'ch{c}'].blending = 'additive'
        viewer.layers[f'ch{c}'].colormap = colormap[c % len(colormap)]
        

def show_sample(sample: dict,
                colormap: tuple = ('green', 'red', 'blue'),
                scale: tuple = (1, 1, 1, 1),
                rescale_intensity: bool = True):

    viewer = napari.Viewer(ndisplay=3)
    if 'image' in sample.keys():

        img = sample['image']
        n_dims = len(img.shape)

        # make img 5D
        if n_dims == 3:
            img_np = np.expand_dims(np.expand_dims(img, axis=0), axis=-1)
        elif n_dims == 4:
            img_np = np.expand_dims(img, axis=-1)
        elif d_dims == 5:
            img_np = img
        else:
            print(f'Warning: cannot decide about the image shape {img.shape}')
            return

        # display individual channels
        for c in range(img_np.shape[0]):

            im = img_np[c]

            if rescale_intensity:
                pL, pH = np.percentile(im, (.5, 99.5))
                im = exposure.rescale_intensity(im, in_range=(pL, pH))

            im = np.moveaxis(im, -2, 0)
            im = np.moveaxis(im, -1, 0)

            print(im.shape)

            viewer.add_image(im,
                             name=f'image_channel{c}',
                             scale=scale,
                             interpolation3d='nearest',
                             blending='additive',
                             colormap=colormap[c % len(colormap)])

    if 'mask' in sample.keys():

        mask = sample['mask']
        n_dims = len(mask.shape)

        # make mask 4D
        if n_dims == 3:
            mask_np = np.expand_dims(mask, axis=-1)
        elif d_dims == 4:
            mask_np = mask
        else:
            print(f'Warning: cannot decide about the image shape {mask.shape}')
            return

        # display mask channels
        im = np.moveaxis(mask_np, -2, 0)
        im = np.moveaxis(im, -1, 0)

        viewer.add_labels(im,
                          name=f'mask',
                          scale=scale,
                          blending='additive')

    if ('keypoints' in sample.keys()) and (len(sample['keypoints']) > 0):

        keypoints = sample['keypoints']
        np_key = np.array(keypoints)

        if np_key.shape[1] == 3:
            np_key = np.pad(np_key, ((0, 0), (1, 0)))

        np_key[:, 1:] = np.roll(np_key[:, 1:], 1, axis=1)

        viewer.add_points(np_key,
                          name=f'keypoints',
                          scale=scale,
                          blending='additive',
                          size=1)


    if ('bboxes' in sample.keys()) and (len(sample['bboxes']) > 0):

        bboxes = sample['bboxes']

        np_bboxes = []
        for bbox in bboxes:

            np_bbox = np.array(bbox)
            #print(np_bbox.shape)

            if np_bbox.shape[1] == 3:
                np_bbox = np.pad(np_bbox, ((0, 0), (1, 0)))

            #print(np_bbox.shape, np_bbox[:3])

            np_bbox[:, 1:] = np.roll(np_bbox[:, 1:], 1, axis=1)

            np_bboxes.append(np_bbox)

        viewer.add_shapes(np_bboxes,
                          name=f'bboxes',
                          scale=scale,
                          blending='additive',
                          shape_type='rectangle',
                          edge_width=0.2,
                          opacity=0.1,

        )



        
           
def concatenate_samples(samples: list,
                        axis: int = 1) -> dict:
    
    imgs, masks = [], []
    for sample in samples:
        imgs.append(sample['image'])
        masks.append(sample['mask'])
        
    return {'image': np.concatenate(imgs, axis=axis),
            'mask': np.concatenate(masks, axis=axis)}
    

        