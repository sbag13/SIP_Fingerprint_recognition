from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy

PATH = "./fingerprints/"


def file_names(directory=PATH, file_ends_with=".tif"):
    """returns list with names of all .tif files in fingerprints directory"""
    file_names = [file_name for file_name in listdir(directory) if isfile(
        join(directory, file_name)) and file_name.endswith(file_ends_with)]
    return file_names
