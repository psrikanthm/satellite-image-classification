from datetime import datetime
import time 
dt = datetime.today()

TRAIN_TIF = "../data/train-tif-v2/"
TRAIN_JPG = "../data/train-jpg/"
XTRAIN_TIF = "../data/x_train_tif.npy"
XTRAIN_JPG = "../data/x_train_jpg.npy"
YTRAIN_CSV = "../data/train.csv"

TEST_TIF = "../data/test-tif-v2/"
TEST_JPG = "../data/test-jpg/"
XTEST_TIF = "../data/x_test_tif.npy"
XTEST_JPG = "../data/x_test_jpg.npy"
XTEST_FILES = "../data/test_filenames.npy"

YTEST_CL = "../data/y_test_cl.npy"
YTEST_WR = "../data/y_test_wr.npy"
YTEST_RL = "../data/y_test_rl.npy"
PRED_FILE = "../data/prediction_" + str(int(time.mktime(dt.timetuple()))) + ".csv"

common_labels = ["primary", "cultivation", "water", "habitation", "agriculture", "road", "bare_ground"]
weather_labels = ["cloudy", "partly_cloudy", "haze", "clear"]
rare_labels = ['slash_burn', 'blooming', 'conventional_mine', 'artisinal_mine', 'blow_down', 'selective_logging']
