import numpy as np
import scipy.ndimage
from .image_enhance import image_enhance


def enhance_image(image_with_path):
    print('Loading image.');
    img = scipy.ndimage.imread(image_with_path)

    if len(img.shape) > 2:
        # img = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
        img = np.dot(img[..., :3], [0.299, 0.587, 0.114])

    rows, cols = np.shape(img)
    aspect_ratio = np.double(rows) / np.double(cols)

    new_rows = 350  # randomly selected number
    new_cols = new_rows / aspect_ratio

    # img = cv2.resize(img,(new_rows,new_cols));
    img = scipy.misc.imresize(img, (np.int(new_rows), np.int(new_cols)))

    return image_enhance(img)
