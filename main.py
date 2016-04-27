# RUNNING COMMAND:
# python main.py [uu/mm/pcc/mf] [cosine/dot] [mean/weight] [k] [output_filepath]

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
from colb_ranking import get_colbrk_train, get_colbrk_pred, svm, colb_ranking_output, analyze


def main(argv):
    # TIMER
    t0 = time.time()

    # PARAMETERS
    model_arg = sys.argv[1]         # [uu|mm|pcc|pmf]
    sim_arg = sys.argv[2]           # [dot|cosine]
    weight_arg = sys.argv[3]        # [mean|weight]
    k = int(sys.argv[4])            # value of k in knn
    latent_num = int(sys.argv[5])   # value of latent factor num in pmf
    output_filepath = sys.argv[6]   # output filepath
    dev_filepath = '../../resources/HW4_data/test.csv'
    test_filepath = '../../resources/HW4_data/test.csv'
    train_filepath = '../../resources/HW4_data/train.csv'
    dev_query_filepath = '../../resources/HW4_data/dev.queries'
    test_query_filepath = '../../resources/HW4_data/test.queries'

    # DATA PREPROCESSING
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
    print time.time() - t0, "seconds wall time"
    return

    # ==========/ Collaborative Ranking / ========== #
    train_v_list, train_y_list = get_colbrk_train(trainM, U, V, five_star_um_dic, one_star_um_dic)
    analyze(train_y_list, obsv_rating_num, five_star_um_dic, one_star_um_dic)
    dev_v_list = get_colbrk_pred(pred_uid_list, pred_mid_list, U, V)  # the vector list to predict
    dev_pred_res_list = svm(train_v_list, train_y_list, dev_v_list)
    colb_ranking_output(dev_pred_res_list, output_filepath)

    print "END!"
    print time.time() - t0, "seconds wall time"

if __name__ == '__main__':
    main(sys.argv[1:])