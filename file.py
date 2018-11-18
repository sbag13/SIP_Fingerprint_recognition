from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy

PATH = "./fingerprints/"


def file_names():
    """returns list with names of all .tif files in fingerprints directory"""
    file_names = [file_name for file_name in listdir(PATH) if isfile(
        join(PATH, file_name)) and file_name.endswith(".tif")]
    return file_names


def feature_vector(file_path):
    """returns feature vector extracted from given image file"""
    image = Image.open(file_path)
    image_array = numpy.array(image).ravel()
    return image_array


def get_samples():
    samples = []
    for f in file_names():
        samples.append(feature_vector(PATH + f))
    samples_np = numpy.array(samples)
    return samples_np
