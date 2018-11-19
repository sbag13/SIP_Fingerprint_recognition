from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy

PATH = "./fingerprints/"


def file_names(scan_number=""):
    """returns list with names of all .tif files in fingerprints directory"""
    file_names = [file_name for file_name in listdir(PATH) if isfile(
        join(PATH, file_name)) and file_name.endswith(scan_number + ".tif")]
    return file_names


def feature_vector(file_path):
    """returns feature vector extracted from given image file"""
    return image_array(file_path).ravel()


def image_array(file_path):
    """returns 2D image array"""
    return numpy.array(Image.open(file_path))


def get_1D_samples(scan_number=""):
    samples = []
    for f in file_names(scan_number):
        samples.append(feature_vector(PATH + f))
    return numpy.array(samples)


def get_2D_samples():
    samples = []
    for f in file_names():
        samples.append(image_array(PATH + f))
    return numpy.array(samples)
