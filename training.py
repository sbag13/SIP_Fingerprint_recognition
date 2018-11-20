from som import SOM
from feature_extractor import extract_features
from file import file_names

path = "./fingerprints/"
samples = []
for file in file_names(scan_number="1"):
    samples.append(extract_features(image_path=path + file, vector_size=32))

print(len(samples))

som = SOM(15, 15, dim=2048, n_iterations=100)
som.train(samples)

# read different image not used for training
path_to_image = "./fingerprints/012_3_2.tif"
test_vector = extract_features(image_path=path_to_image, vector_size=32)

# indexes of original samples
print(som.map_vects([samples[1]]))
print(som.map_vects([test_vector]))

