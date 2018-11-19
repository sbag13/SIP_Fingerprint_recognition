from som import SOM
from file import get_1D_samples, file_names
from matplotlib import pyplot as plt

samples = get_1D_samples(scan_number="1")
scan_names = file_names(scan_number="1")

som = SOM(15, 15, dim=samples.shape[1], n_iterations=1)
som.train(samples)

# indexes of original samples
mapped = som.map_vects(samples)
print(mapped)