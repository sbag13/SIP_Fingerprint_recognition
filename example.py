from training import *

initialize_som(5,25,25)    # initialize with max 10 iterations
train(3)                    # train 3 iterations
train(3)                    # train 3 iterations, only 2 will execute
save_som('10_25_25')        # save to directory '10_25_25'
# load_som('10_25_25')
test_accuracy()