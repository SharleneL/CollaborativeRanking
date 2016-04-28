# RUNNING COMMAND:
# python main.py [uu|mm|pcc|pmf|colbrk-svm|colbrk-lr] [dot|cosine] [mean|weight] [k] [latent_factor_num] [output_filepath]
# *LR example* python main.py colbrk-lr dot mean 5 100 ../../code_output/hw6/lr_100

__author__ = 'luoshalin'

from scipy import sparse
import os
import sys
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from scipy import spatial
import heapq
from numpy import linalg as LAn
import time
from scipy.sparse import *
from scipy import *

from preprocess import get_trainM, get_qmM, get_pred_lists
from user_sim import get_user_user_pred, uu_output
from movie_sim import get_movie_movie_pred, mm_output
from matrix_factorization import pmf_train, pmf_pred, pmf_output
from colb_ranking import get_colbrk_train, get_colbrk_pred, svm_train, svm_predict, colb_ranking_output, analyze
from lr.lr import lr_train_param, lr_predict


def main(argv):
    # TIMER
    t0 = time.time()

    # ========== / PARAMETERS / ========== #
    # --- / CMD PARAMS / --- #
    model_arg = sys.argv[1]         # [uu|mm|pcc|pmf|colbrk-svm|colbrk-lr]
    sim_arg = sys.argv[2]           # [dot|cosine]
    weight_arg = sys.argv[3]        # [mean|weight]
    k = int(sys.argv[4])            # value of k in knn
    latent_num = int(sys.argv[5])   # value of pmf latent factor num
    output_filepath = sys.argv[6]   # output filepath

    # --- / FILEPATH PARAMS / --- #
    dev_filepath = '../../resources/HW4_data/test.csv'
    test_filepath = '../../resources/HW4_data/test.csv'
    train_filepath = '../../resources/HW4_data/train.csv'
    dev_query_filepath = '../../resources/HW4_data/dev.queries'
    test_query_filepath = '../../resources/HW4_data/test.queries'

    # --- / PMF PARAMS - IF USED / --- #
    pmf_lmd = 0.01
    pmf_step = 0.0001
    pmf_itr = 200
    pmf_threshold = 10e-3  # stopping criteria
    # pmf_threshold = 0.2  # stopping criteria - test

    # --- / LOGISTIC REGRESSION PARAMS - IF USED / --- #
    lr_feature_num = latent_num
    lr_lmd = 0.01                          # lambda
    lr_alpha = 0.001                       # learning rate
    lr_threshold = 10E-6                   # stopping criteria; changing rate of log likelihood
    # lr_threshold = 10E-3                   # stopping criteria - test
    lr_gd_method = 'bsga'                  # gradient ascend method: fill in with [sga] or [bsga]
    lr_batch_size = 200

    # ========== / DATA PREPROCESSING / ========== #
    # read train data -> preprocessing into vectors -> imputation
    trainM, five_star_um_dic, one_star_um_dic, obsv_rating_num = get_trainM(train_filepath)  # get a sparse M as training set, <user, movie>[score]
    # save target <query, movie> pairs to be predicted into a matrix
    mu_list = []
    qmM = get_qmM(dev_filepath, mu_list)
    pred_uid_list, pred_mid_list = get_pred_lists(dev_filepath)

    # ==========/ EXP 1(uu) /========== #
    if model_arg == 'uu':
        qid_set = set(find(qmM)[0])  # row - queries
        qid_list = list(qid_set)
        # run user-user similarity algo
        uu_pred_dic = get_user_user_pred(qid_list, trainM, k, sim_arg, weight_arg)
        # write to result
        uu_output(uu_pred_dic, mu_list, dev_filepath, output_filepath)

    # ==========/ EXP 2(mm) /========== #
    if model_arg == 'mm':
        # run movie-movie similarity algo
        mm_pred_dic = get_movie_movie_pred(qmM, trainM, k, sim_arg, weight_arg, model_arg)
        mm_output(mm_pred_dic, mu_list, dev_filepath, output_filepath)

    # ==========/ EXP 3(pcc) /========== #
    if model_arg == 'pcc':
        mm_pred_dic = get_movie_movie_pred(qmM, trainM, k, sim_arg, weight_arg, model_arg)
        mm_output(mm_pred_dic, mu_list, dev_filepath, output_filepath)

    # ==========/ EXP 4(pmf) /========== #
    if model_arg == 'pmf':
        U, V = pmf_train(trainM, latent_num)
        pmf_pred_res_list = pmf_pred(U, V, pred_uid_list, pred_mid_list)
        pmf_output(pmf_pred_res_list, output_filepath)
        print "pmf END!"

    # ==========/ Collaborative Ranking / ========== #
    if model_arg == 'colbrk-svm' or 'colbrk-lr':
        # GENERATE pmf U, V
        U, V = pmf_train(trainM, latent_num, pmf_lmd, pmf_step, pmf_itr, pmf_threshold)
        # GENERATE TRAINING INPUT
        train_v_list, train_y_list, lr_train_M, lr_train_stars_M = get_colbrk_train(trainM, U, V, five_star_um_dic, one_star_um_dic)  # contain a list of v features(list of list) & a list of labels(list of number)
        # DATASET STATISTICS
        # analyze(train_y_list, obsv_rating_num, five_star_um_dic, one_star_um_dic)
        # GENERATE PREDICT INPUT
        print 'line91'
        dev_v_list = get_colbrk_pred(pred_uid_list, pred_mid_list, U, V)  # the vector list to predict

        # MODEL#1: svm (train & predict)
        dev_M = sparse.csr_matrix(np.array(dev_v_list))          # <#datapoint, #latentfactor> convert list of list(vector) to csr matrix - each row is a vector
        if model_arg == 'colbrk-svm':
            print 'main svm line 113'
            dev_pred_res_list = svm(train_v_list, train_y_list, dev_v_list)
            print 'main svm line 115'
            colb_ranking_output(dev_pred_res_list, output_filepath)
            print 'main svm line 117'

            W = svm_train(train_v_list, train_y_list, dev_v_list)
            svm_pred_res_list = svm_predict(dev_M, W)
            colb_ranking_output(svm_pred_res_list, output_filepath)

        elif model_arg == 'colbrk-lr':
            # MODEL#2: lr
            # lr train
            print "line93"
            lr_eval_M = lr_train_M                  # lr_train_M and eval_M are both sparseM, each line is a vector; use the trainM as the evalM
            lr_eval_stars_M = lr_train_stars_M      # lr_stars_train_M and stars_eval_M are both sparseM, each line is a vector; use the trainM as the evalM

            print "line103"
            W_org = np.ones(shape=(lr_feature_num, 2)) * float(1)/lr_feature_num  # class_num = 2 (2 categories)
            W = lr_train_param(lr_train_M, lr_train_stars_M, W_org, lr_eval_M, lr_eval_stars_M, lr_lmd, lr_alpha, lr_threshold, lr_gd_method, lr_batch_size)  # eval_M is the evaluation dataset
            print "line105"

            print "line110"
            lr_pred_res_list = lr_predict(dev_M, W)
            # output - save to file
            colb_ranking_output(lr_pred_res_list, output_filepath)

    print "END!"
    print time.time() - t0, "seconds wall time"

if __name__ == '__main__':
    main(sys.argv[1:])