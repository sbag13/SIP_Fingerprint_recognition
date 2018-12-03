from som import SOM
from feature_extractor import extract_features
from file import file_names
import numpy as np
from cropp_image import trim
import sys
import os
import pickle
import json

original_images_path = './fingerprints/'
cropped_images_path = './cropped_fingerprints/'
output_extracted_vector_file = 'extracted_vectors.txt'
output_properties_file = 'som_properties.txt'
output_weightages = './weightages'
output_locations = './locations'
output_session = './session.ckpt'

samples = []
samples_to_train = []
samples_to_test = []

# ------------------ preparing data to train and test --------------------------


def cropp_images():
    if len(os.listdir(cropped_images_path)) == 0:
        print('Cropping images.')
        file_names_list = file_names(
            directory='./fingerprints/', file_ends_with='.tif')
        for i, file in enumerate(file_names_list):
            print('Cropping %d/%d' % (i, len(file_names_list)))
            trim(input_path=original_images_path +
                 file).save(cropped_images_path + file.replace('.tif', '.png'))
    else:
        print('Images are cropped.')


def load_feature_vec():
    global samples
    if not os.path.isfile(output_extracted_vector_file):
        cropp_images()
        print('Generating feature vectors.')
        file_names_list = file_names(
            directory='./cropped_fingerprints/', file_ends_with='.png')
        for i, file in enumerate(file_names_list):
            print('Extracting %d/%d' % (i + 1, len(file_names_list)), end='\r')
            samples.append(extract_features(
                image_path=cropped_images_path + file, vector_size=16))
        print('Saving feature vectors into: ' + output_extracted_vector_file)
        np.savetxt(output_extracted_vector_file, samples)
    else:
        print('Loading feature vectors from file:' +
              output_extracted_vector_file)
        with open(output_extracted_vector_file) as textFile:
            samples = [line.split(' ') for line in textFile]


def load_samples():
    global samples_to_train
    global samples_to_test

    if len(samples) == 0:
        load_feature_vec()

    for x in range(len(samples)):
        if x % 8 == 0:
            continue
        else:
            samples_to_train.append(samples[x])

    samples_to_train = np.array([np.array(el).astype(
        np.float32) for el in samples_to_train])  # convert string to floats
    samples_to_test = np.array([np.array(el).astype(np.float32)
                                for el in np.array(samples)[::8]])

# ------------------ training network and testing --------------------------


som = None


def initialize_som(n=30, m=30, iterations=100):
    global som
    print('Initialising SOM.')
    som = SOM(n, m, dim=1024, n_iterations=iterations)


def train(iterations):
    global som
    if len(samples_to_train) == 0:
        load_samples()
    if som == None:
        initialize_som()
    print('Training SOM.')
    som.train(samples_to_train, iterations)

def save_som():
    properties = {}
    properties["trained_iterations"] = som._trained_iterations
    properties["n_iterations"] = som._n_iterations
    properties["m"] = som._m
    properties["n"] = som._n
    properties["sigma"] = som._sigma
    properties["alpha"] = som._alpha
    properties["dim"] = som._dim
    properties["trained"] = som._trained
    np.save(output_weightages, som._weightages)
    np.save(output_locations, som._locations)
    with open(output_properties_file, 'w') as output:
        output.write(json.dumps(properties))


def load_som():
    global som
    weightages = np.load(output_weightages + ".npy")
    locations = np.load(output_locations + ".npy")
    with open(output_properties_file) as input:
        properties = json.load(input)
    som = SOM(properties["m"], properties["n"], properties["dim"], properties["n_iterations"],
              properties["alpha"], properties["sigma"], properties["trained_iterations"], weightages, locations, properties["trained"])


# print('Mapping test samples:')
#  print(som.map_vects(samples_to_test))

# test 1st print
# first_print_samples = np.array([np.array(el).astype(np.float32) for el in samples[:9]])
# print(som.map_vects(first_print_samples))

# TODO
# loop status prints
# saving loading session
