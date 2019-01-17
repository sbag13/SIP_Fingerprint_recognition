import cv2
import numpy as np
from scipy.misc import imread
import _pickle as pickle
import os
from matplotlib import pyplot as plt

def extract_features(image_path, algorithm, vector_size=32):
    image = imread(image_path, mode="L")  # L - grey-scale
    try:
        if algorithm == "kaze":
            alg = cv2.KAZE_create()
            needed_size = (vector_size * 64)
        elif algorithm == "sift":
            alg = cv2.xfeatures2d.SIFT_create()
            needed_size = (vector_size * 128)
        elif algorithm == "surf":
            needed_size = (vector_size * 64)
            alg = cv2.xfeatures2d.SURF_create()
        elif algorithm == "orb":
            needed_size = (vector_size * 32)
            alg = cv2.ORB_create()
        else:
            alg = cv2.KAZE_create()
            needed_size = (vector_size * 64)

        # Dinding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them.
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print('Error: ', e)
        return None

    # printing keypoints
    img2 = cv2.drawKeypoints(image, kps, None, color=(0,255,0), flags=0)
    # plt.imshow(img2), plt.show()
    file_name = image_path.split("/")[2]
    cv2.imwrite("./keypoints_images/" + algorithm + "_" + str(vector_size) + "/" + file_name, img2)

    return dsc


def batch_extractor(images_path, algorithm, pickled_db_path="features.pck"):
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]

    result = {}
    for f in files:
        print('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        result[name] = extract_features(f, algorithm)

    # saving all our feature vectors in pickled file
    with open(pickled_db_path, 'w') as fp:
        pickle.dump(result, fp)
