import cv2
import numpy as np

def binarize_image(image_path):
    im_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    # cv2.imwrite('test.png', im_bw) # save image

    return im_bw


def invert_colors(image):
    return cv2.bitwise_not(image)


def skeletonize_image(image):
    size = np.size(image)
    skeleton = np.zeros(image.shape, np.uint8)

    ret, img = cv2.threshold(image, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    return skeleton


def extract_features(image_path, corners_number=32):
    gray = skeletonize_image(invert_colors(binarize_image(image_path)))
    corners = cv2.goodFeaturesToTrack(gray, corners_number, 0.2, 5)
    corners = np.squeeze(np.int0(corners))

    result = np.array(corners[corners[:, 0].argsort()]).flatten()
    print(result.shape)
    return result


def show_detected_corners(img, corners):
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img, (x, y), 5, (0, 255, 0), 2)

    cv2.imshow("Harris corner", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# show_detected_corners(cv2.imread("fingerprints2/101_1.tif"), extract_features("fingerprints2/101_1.tif"))

