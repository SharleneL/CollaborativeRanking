__author__ = 'luoshalin'

import numpy as np
from preprocess import get_review_list, print_analysis
from featureExtract import gen_libsvm_file, get_train_feature_M, get_test_feature_M
from lr import lr_train_param, lr_predict
from evaluate import cal_accuracy, cal_rmse, print_pred_res


def main():
    # PARAMETERS - file path
    input_fpath = '../../resources/data/yelp_reviews_train.json'        # file path for train set
    test_input_fpath = '../../resources/data/yelp_reviews_test.json'    # file path for test/dev set
    output_fpath = 'pred_result.out'                                    # file path to save the predict results for test/dev set
    stopword_fpath = '../../resources/data/stopword.list'               # file path for stop word list

    # PARAMETERS - logistic regression
    feature_method = 'ctf'              # feature extraction method: fill in with [ctf] or [df]
    feature_num = 4000                  # number of top features
    class_num = 5                       # classification category number
    lmd = 0.01                          # lambda
    alpha = 0.001                       # learning rate
    threshold = 10E-6                   # stopping criteria; changing rate of log likelihood
    gd_method = 'bsga'                  # gradient ascend method: fill in with [sga] or [bsga]
    batch_size = 200                    # batch size for batched gradient ascend
    eval_data_frac = 0.04               # percentage of data to be splitted out as evaluation data


    print 'PROGRAM STARTS!'
    # ---------- / GENERATE MIDDLE FILES - start / ---------- #
    # TASK1: PREPROCESS
    # each row of stars_M is a vector.T of corresponding review. only one col in the vector is '1'
    review_list, stars_list, total_data_amt, test_review_list, test_stars_list, test_total_data_amt = get_review_list(stopword_fpath, input_fpath, test_input_fpath)  # stars_M: <datanum, classnum:5>
    # print the analytical results
    print_analysis(review_list)

    # TASK2: FEATURE DESIGN
    # generate review_feature libsvm files & feature_dict files
    libsvm_file_path, test_libsvm_file_path = gen_libsvm_file(review_list, stars_list, test_review_list, feature_method, feature_num)  # get the file for useful features
    # ---------- / GENERATE MIDDLE FILES - end / ---------- #


    # ---------- / USE MIDDLE FILE - start / ---------- #
    # === file path === #
    # libsvm_file_path = '../feature_data/test/4000/ctf_libsvm_train.out'  # hard coded filepath - for efficiency
    # test_libsvm_file_path = '../feature_data/test/4000/ctf_libsvm_test.out'  # hard coded filepath - for efficiency
    # === data amount === #
    # total_data_amt = 1255353          # train
    # total_data_amt = 157010           # dev
    # total_data_amt = 156901             # test
    # ---------- / USE MIDDLE FILE - end / ---------- #

    # generate features & useful matrix
    train_M, eval_M, train_stars_M, eval_stars_M, eval_stars_list = get_train_feature_M(libsvm_file_path, total_data_amt, eval_data_frac, feature_num, class_num)
    test_M = get_test_feature_M(test_libsvm_file_path, feature_num)

    # TASK3.1: LR TRAIN
    W_org = np.ones(shape=(feature_num, class_num)) * float(1)/feature_num
    W = lr_train_param(train_M, train_stars_M, W_org, eval_M, eval_stars_M, lmd, alpha, threshold, gd_method, batch_size)  # eval_M is the evaluation dataset

    # TASK3.2: EVALUATION
    # on the evaluation set
    pred_list_hard, pred_list_soft = lr_predict(eval_M, W)
    # on the test/dev set
    test_pred_list_hard, test_pred_list_soft = lr_predict(test_M, W)
    # output test/dev evaluation results to file
    print_pred_res(test_pred_list_hard, test_pred_list_soft, output_fpath)

    # calculate accuracy
    accuracy = cal_accuracy(eval_stars_list, pred_list_hard)
    print 'Accuracy: ' + str(accuracy)
    # calculate rmse
    rmse = cal_rmse(eval_stars_list, pred_list_soft)
    print 'RMSE: ' + str(rmse)
    print 'PROGRAM FINISHED!'

if __name__ == '__main__':
    main()