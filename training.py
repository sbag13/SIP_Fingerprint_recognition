from itertools import product
from som import SOM
from feature_extractor import extract_features
from file import file_names
import numpy as np
from cropp_image import trim
import os
import json
import math

np.set_printoptions(threshold=np.nan)

original_images_path = './fingerprints/'
cropped_images_path = './cropped_fingerprints/'
output_extracted_vector_file = 'extracted_vectors.txt'
output_properties_file = '/som_properties.txt'
output_weightages = '/weightages'
output_locations = '/locations'

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
    global som
    print('Initialising SOM. N iterations %d' % max_iterations)
    som = SOM(n, m, dim=1024, n_iterations=max_iterations)


def train(iterations=1):
    """
    Makes given number of training iterations,
    may be less than max_iterations of SOM
    Cant train more than max_iterations of SOM
    Initialize som and load samples if needed
    """
    global som
    global samples_to_train
    if len(samples_to_train) == 0:
        load_samples()
    if som == None:
        initialize_som()
    print('Training SOM.')
    som.train(samples_to_train, iterations)


def map_training_vects():
    """
    fill trained_fingerprints_locations with numbers of fingerprints
    0 - no fingerprint
    """
    global som
    global trained_fingerprints_locations
    global samples_to_train

    props = som.get_properties()
    n = props["n"]
    m = props["m"]

    if som is None:
        print("No som loaded.")
        return
    if len(samples_to_train) == 0:
        load_samples()

    trained_fingerprints_locations = np.zeros((n, m), dtype=int)
    locations = som.map_vects(samples_to_train)
    for i in range(len(locations)):
        trained_fingerprints_locations[locations[i][0]][locations[i][1]] = i / 7 + 1


def test_accuracy():
    """
    checks accuracy of mapping test samples
    if did not hit precisely, find in neighbourhood
    """
    global som
    global trained_fingerprints_locations
    global samples_to_test

    props = som.get_properties()
    n = props["n"]
    m = props["m"]

    if som is None:
        print("No som loaded.")
        return
    if len(samples_to_test) == 0:
        load_samples()
    if trained_fingerprints_locations is None:
        map_training_vects()

    locations = som.map_vects(samples_to_test)
    hits = 0
    for i in range(len(samples_to_test)):
        guess_fingerprint_number = trained_fingerprints_locations[locations[i][0]][locations[i][1]]
        if guess_fingerprint_number == i + 1:
            print("GREAT HIT! sample number: %d" % (i + 1))
            hits += 1
        else:
            print("missed. test_number:%d, got_number:%d" %
                  (i + 1, guess_fingerprint_number))
            if guess_fingerprint_number == 0:
                closest_fingerprint_number = find_closest_fingerprint(
                    locations[i], n, m)
                if all(element == i + 1 for element in closest_fingerprint_number):
                    print("SMALL HIT!")
                    hits += 1
                elif any(element == i + 1 for element in closest_fingerprint_number):
                    print("THAT WAS CLOSE!")
                else:
                    print("closest numbers")
                print(closest_fingerprint_number)

    print("Score: %d/%d" % (hits, len(samples_to_test)))


def find_closest_fingerprint(location, n, m):
    """
    finds closest fingerprints in trained_fingerprints_locations to given location
    may return few items with equal distance
    brutal-force, unefficient
    """
    global trained_fingerprints_locations
    min_dist = n * m
    closest_targets = []
    for i, j in product(range(n), range(m)):
        if trained_fingerprints_locations[i][j] != 0:
            distance = math.sqrt(
                pow(location[0] - i, 2) + pow(location[1] - j, 2))
            if distance < min_dist:
                min_dist = distance
                closest_targets.clear()
                closest_targets.append(trained_fingerprints_locations[i][j])
            elif distance == min_dist:
                closest_targets.append(trained_fingerprints_locations[i][j])
    return closest_targets


def find_locations_of_fingerprint(finger_print_number, n=30, m=30):
    global trained_fingerprints_locations

    if trained_fingerprints_locations == None:
        map_training_vects()

    locations = []
    for i, j in product(range(n), range(m)):
        if trained_fingerprints_locations[i][j] == finger_print_number:
            locations.append([i, j])
    return locations


def check_presence_of_all_fingerprints():
    """
    checks if all fingerprints are present in trained_fingerprints_locations
    sometimes few are mapped to the same neuron, so may be overwritten
    """
    global som
    global samples_to_test
    if som is None:
        print("No som loaded.")
        return
    if len(samples_to_test) == 0:
        load_samples()

    locations = []
    for i in range(len(samples_to_test)):
        locations = find_locations_of_fingerprint(i + 1)
    if all(location for location in locations):
        print("All here")


def save_som(net_name):
    """
    save som to directory net_name
    """
    global som

    if not os.path.exists(net_name):
        os.makedirs(net_name)
    np.save(net_name + output_weightages, som._weightages)
    np.save(net_name + output_locations, som._locations)
    with open(net_name + output_properties_file, 'w') as output:
        output.write(json.dumps(som.get_properties()))


def load_som(net_name):
    """
    loads som from directory net_name
    """
    global som

    if not os.path.exists(net_name):
        print("No such som")
        return

    weightages = np.load(net_name + output_weightages + ".npy")
    locations = np.load(net_name + output_locations + ".npy")
    with open(net_name + output_properties_file) as input:
        properties = json.load(input)
    som = SOM(properties["m"], properties["n"], properties["dim"], properties["n_iterations"],
              properties["alpha"], properties["sigma"], properties["trained_iterations"], weightages, locations, properties["trained"])


def print_properties():
    """
    prints properties of loaded SOM
    """
    global som
    if som is not None:
        print(som.get_properties())
    else:
        print("No som initialized!.")


def clear():
    """
    clear environmental variables
    """
    global samples
    global samples_to_test
    global samples_to_train
    global som
    global trained_fingerprints_locations
    samples = []
    samples_to_test = []
    samples_to_train = []
    som = None
    trained_fingerprints_locations = None
