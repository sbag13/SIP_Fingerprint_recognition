from training import *

rm_features()
load_feature_vec("kaze")
initialize_som(4,30,30,2048)    # initialize with max 10 iterations
# train(10)                    # train 3 iterations
train(4)                    # train 3 iterations, only 2 will execute
# save_som('10_30_30_surf_vecsize32')        # save to directory '10_25_25'
# # # load_som('10_25_25')
test_accuracy()
