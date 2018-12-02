from som import SOM
from feature_extractor import extract_features
from file import file_names
import numpy as np
from cropp_image import trim
import os

original_images_path = './fingerprints/'
cropped_images_path = './cropped_fingerprints/'
output_file = 'extracted_vectors.txt'
samples = []

# ------------------ preparing data to train and test --------------------------
if len(os.listdir(cropped_images_path)) == 0:
    print('Cropping images.')
    for file in file_names(directory='./fingerprints/', file_ends_with='.tif'):
        trim(input_path=original_images_path + file).save(cropped_images_path + file.replace('.tif', '.png'))
else:
    print('Images are cropped.')

if not os.path.isfile(output_file):
    print('Generating feature vectors.')
    for file in file_names(directory='./cropped_fingerprints/', file_ends_with='.png'):
        samples.append(extract_features(image_path=cropped_images_path + file, vector_size=16))
    print('Saving feature vectors into: ' + output_file)
    np.savetxt(output_file, samples)
else:
    print('Loading feature vectors from file:' + output_file)
    with open(output_file) as textFile:
        samples = [line.split(' ') for line in textFile]

samples_to_train = []
for x in range(len(samples)):
    if x % 8 == 0:
        continue
    else:
        samples_to_train.append(samples[x])

samples_to_train = np.array([np.array(el) for el in samples_to_train]).astype(np.float32)
samples_to_test = np.array(samples)[::8]

# ------------------ training network and testing --------------------------
print('Initialising SOM.')
som = SOM(30, 30, dim=1024, n_iterations=1)
print('Training SOM.')
som.train(samples_to_train)

print('Mapping test samples:')
print(som.map_vects(samples_to_test))

