import numpy as np
from skimage.transform import resize


def resize_complex(complex_image, *args, **kwargs):
    real = np.real(complex_image)
    real = resize(real, *args, **kwargs)
    imag = np.imag(complex_image)
    imag = resize(imag, *args, **kwargs)
    return real + (1j*imag)


def resize_images(images, new_height, new_width):
    '''
    Inp: [Batch, H, W, ...Other Dims...]
    NB/ Only downsamples H and W. Not Time.
    '''
    N = images.shape[0]
    new_images = []
    for i in range(N):
        this_image = images[i:i+1]
        this_shape = this_image.shape
        new_shape = list(this_shape[:])
        new_shape[1] = new_height
        new_shape[2] = new_width
        new_images.append(
            resize_complex(
                this_image,
                new_shape,
                anti_aliasing=True
            )
        )
    new_images = np.concatenate(new_images, axis=0)
    return new_images


def central_crop(image, t, images_2d=False, auto_shape=True):
    '''
    Works for Batch, Time, Width, Height NB/:
     - Batch and Time dimensions are optional
    '''
    s = image.shape # [256, 256, S]

    images = np.expand_dims(image, axis=0) # [1, 256, 256, S]
    h = s[0]


    t_ = int((h-t)/2)
    new_images = crop_to_bounding_box(
        images,
        t_, t_, t, t
    )

    new_images = np.squeeze(new_images, axis=0)
    return new_images


def crop_to_bounding_box(image, oh: int, ow: int, th: int, tw: int):

    return image[
        :,
        oh:oh+th,
        ow:ow+tw,
        :
    ]
