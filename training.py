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
trained_fingerprints_locations = None


def initialize_som(max_iterations=100, n=30, m=30):
    """
    max_iterations - indicates when som is fully trained
    """
    global som
    print('Initialising SOM.')
    som = SOM(n, m, dim=1024, n_iterations=max_iterations)


def train(iterations):
    """
    Makes given number of training iterations, may be less than max_iterations of som
    Initialize som and load samples if needed
    """
    global som
    if len(samples_to_train) == 0:
        load_samples()
    if som == None:
        initialize_som()
    print('Training SOM.')
    som.train(samples_to_train, iterations)


def print_properties():
    global som
    if som is not None:
        print(som.get_properties())
    else:
        print("No som initialized!.")


def save_som():
    som.save(output_properties_file, output_weightages, output_locations)


def load_som():
    global som
    weightages = np.load(output_weightages + ".npy")
    locations = np.load(output_locations + ".npy")
    with open(output_properties_file) as input:
        properties = json.load(input)
    som = SOM(properties["m"], properties["n"], properties["dim"], properties["n_iterations"],
              properties["alpha"], properties["sigma"], properties["trained_iterations"], weightages, locations, properties["trained"])


def map_training_vects(n=30, m=30):
    global som
    global trained_fingerprints_locations
    global samples_to_train
    if som is None:
        print("No som loaded.")
        return
    if len(samples_to_train) == 0:
        load_samples()

    trained_fingerprints_locations = np.zeros((n, m), dtype=int)
    locations = som.map_vects(samples_to_train)
    for i in range(len(locations)):
        print(locations[i])  # debug
        trained_fingerprints_locations[locations[i][0]][locations[i][1]] = i / 7 + 1


def test_accuracy():
    global som
    global trained_fingerprints_locations
    global samples_to_test
    if som is None:
        print("No som loaded.")
        return
    if len(samples_to_test) == 0:
        load_samples()
    if trained_fingerprints_locations is None:
        map_training_vects()

    locations = som.map_vects(samples_to_test)
    for i in range(len(samples_to_test)):
        idx = trained_fingerprints_locations[locations[i][0]][locations[i][1]]
        if idx == i + 1:
            print("HIT!")
        else:
            print("missed. test_number:%d, got_number:%d" % (i + 1, idx))


initialize_som(1,30,30)
train(100)
map_training_vects(30,30)
print(som._weightages) # lot of NaN's
test_accuracy()

# print('Mapping test samples:')
#  print(som.map_vects(samples_to_test))

# test 1st print
# first_print_samples = np.array([np.array(el).astype(np.float32) for el in samples[:9]])
# print(som.map_vects(first_print_samples))
