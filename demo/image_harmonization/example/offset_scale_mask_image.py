import cv2
import numpy as np

def bounding_box_from_mask(mask):
    """Return the bounding box of the mask.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return (rmin, rmax, cmin, cmax)

def offset_scale_mask_image(mask, image, bg_image, offset, scale):
    """Offset and scale the mask and apply it to the image.
    """
    rmin, rmax, cmin, cmax = bounding_box_from_mask(mask)
    mask = mask[rmin:rmax, cmin:cmax]
    image = image[rmin:rmax, cmin:cmax]
    mask = cv2.resize(mask, None, fx=scale, fy=scale)
    image = cv2.resize(image, None, fx=scale, fy=scale)
    mask = mask.astype(np.float32)
    image = image.astype(np.float32)
    
    image = image.astype(np.uint8)
    image = np.roll(image, offset[0], axis=0)
    image = np.roll(image, offset[1], axis=1)

    fg_image =cv2.multiply(image, mask)
    bg_image = cv2.multiply(bg_image, 1.0 - mask)
    composite_image = fg_image + bg_image
    return composite_image


