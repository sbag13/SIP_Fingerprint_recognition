from som import SOM
from file import get_1D_samples

samples = get_1D_samples()
som = SOM(15, 15, dim=samples.shape[1], n_iterations=5)