from training import *

# rm_features()
# load_feature_vec("harris")
# initialize_som(5,25,25,64)    # initialize with max 10 iterations
# # train(10)                    # train 3 iterations
# train(5)                    # train 3 iterations, only 2 will execute
# save_som('5_25_25_harris_vecsize32')        # save to directory '10_25_25'
# # # # load_som('10_25_25')
# test_accuracy()

start_neural("./harris_32.ckpt")
predict_neural("./harris_32.ckpt")