import os
import tensorflow as tf
import numpy as np


def preprocess_paths(paths):
    if isinstance(paths, list):
        return [os.path.abspath(os.path.expanduser(path)) for path in paths]
    return os.path.abspath(os.path.expanduser(paths)) if paths else None


def random_bbox(img_height=256, img_width=256, margins=0, mask_size=128, random_mask=True):
    """Generate a random tlhw with configuration.

    Args:
        img_height: height of image.
        img_width: width of image.
        margins: margins of mask and image border.
        mask_size: size of mask.
        random_mask: if True, random location. if False, central location.

    Returns:
        tuple: (top, left, height, width)

    """
    if random_mask is True:
        maxt = img_height - margins - mask_size
        maxl = img_width - margins - mask_size
        t = tf.random.uniform(
            [], minval=margins, maxval=maxt, dtype=tf.int32)
        l = tf.random.uniform(
            [], minval=margins, maxval=maxl, dtype=tf.int32)
    else:
        t = (img_height - mask_size) // 2
        l = (img_width - mask_size) // 2
    h = tf.constant(mask_size)
    w = tf.constant(mask_size)
    return (t, l, h, w)


def bbox2mask(bbox, img_height=256, img_width=256, max_delta=32, name='mask'):
    """Generate mask tensor from bbox.

    Args:
        bbox: configuration tuple, (top, left, height, width)
        img_height: height of image.
        img_width: width of image.
        max_delta: max delta of masks.
        name: name of variable scope.

    Returns:
        tf.Tensor: output with shape [1, H, W, 1]

    """

    def npmask(bbox, height, width, delta):
        mask = np.zeros((1, height, width, 1), np.float32)
        h = np.random.randint(delta // 2 + 1)
        w = np.random.randint(delta // 2 + 1)
        mask[:, bbox[0] + h:bbox[0] + bbox[2] - h,
        bbox[1] + w:bbox[1] + bbox[3] - w, :] = 1.
        return mask

    mask = tf.py_function(
        npmask,
        [bbox, img_height, img_width, max_delta],
        tf.float32)
    mask.set_shape([1] + [img_height, img_width] + [1])
    return mask
