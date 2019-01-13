from training import *

# rm_features()
# load_feature_vec("sift")
# initialize_som(5,25,25,2048)    # initialize with max 10 iterations
# train(10)                    # train 3 iterations
# train(5)                    # train 3 iterations, only 2 will execute
# save_som('5_25_25_sift_vecsize16')        # save to directory '10_25_25'
# # # load_som('10_25_25')
# test_accuracy()

start_neural("./kaze_16_model.ckpt")
predict_neural("./kaze_16_model.ckpt")