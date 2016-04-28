__author__ = 'luoshalin'

import numpy as np
from sklearn.svm import SVC
from scipy import *
from scipy import sparse
from scipy.sparse import *


def get_colbrk_train(R, U, V, five_star_um_dic, one_star_um_dic):  # R: sparse matrix  U,V: np matrix
    v_list = []  # vector list
    y_list = []  # label list

    lr_y_row = []   # for lr sparse matrix
    lr_y_col = []   # col=0 if label_y=-1; col=1 if label_y=1
    lr_y_data = []  # =1 if belong to current row, =0 o.w.

    for u in five_star_um_dic.keys():
        if u not in one_star_um_dic:
            continue
        print "|get_colbrk_train| user#" + str(u)
        # for user existing both in five & one star dic
        five_star_mid_list = five_star_um_dic[u]  # list of mids
        one_star_mid_list = one_star_um_dic[u]
        for i in five_star_mid_list:
            for j in one_star_mid_list:
                # calculate <v_ij, y_ij>
                y_1 = 1
                v_1 = np.multiply(U[u], V[i]) - np.multiply(U[u], V[j])  # v_ij, is a matrix
                v_1 = np.asarray(v_1)[0]  # is an array

                # calculate <v_ji, y_ji>
                y_2 = -1
                v_2 = np.multiply(U[u], V[j]) - np.multiply(U[u], V[i])  # v_ij, is a matrix
                v_2 = np.asarray(v_2)[0]  # is an array

                # save <v_1, y_1> and <v_2, y_2> into training data list
                v_list.append(v_1)
                v_list.append(v_2)
                y_list.append(y_1)
                y_list.append(y_2)

                # save label y into sparse matrix list
                # col & data
                if y_1 == -1:
                    lr_y_col.append(0)
                    lr_y_data.append(1)
                else:
                    lr_y_col.append(1)
                    lr_y_data.append(1)
                if y_2 == -1:
                    lr_y_col.append(0)
                    lr_y_data.append(1)
                else:
                    lr_y_col.append(1)
                    lr_y_data.append(1)

    # construct v_M
    lr_v_M = sparse.csr_matrix(np.array(v_list))    # convert vector list into sparse matrix, for lr use
    # construct y_M
    lr_y_row = range(len(lr_y_col))
    lr_y_M = csr_matrix((lr_y_data, (lr_y_row, lr_y_col)), shape=(len(lr_y_row), 2))   # col number = 2 (either -1 or 1)
    return v_list, y_list, lr_v_M, lr_y_M


def get_colbrk_pred(uid_list, mid_list, U, V):
    pred_v_list = []
    for uid, mid in zip(uid_list, mid_list):
        v = np.asarray(np.multiply(U[uid], V[mid]))[0]
        pred_v_list.append(v)
    return pred_v_list


def svm_train(train_v_list, train_y_list, pred_v_list):  # T is the input matrix
    # train
    print "SVM model: begin training..."
    svm_model = SVC()
    svm_model.fit(train_v_list, train_y_list)
    print "### SVM model: end training! ###\n"
    # predict
    print "Begin predicting SVM model..."
    pred_res_list = svm_model.predict(pred_v_list)
    print "### SVM model: end predicting! ###\n"
    # return pred_res_list
    return svm_model.coef_


def svm_predict(x_M, W):
    p_v = W.T[1] - W.T[0]                   # np array, <1, #latentfeature>
    res_M = np.dot(x_M.toarray(), p_v.T)    # <#data point, #latent factor> * <#latentfeature, 1>
    return res_M.tolist()


def colb_ranking_output(res_list, output_filepath):
    with open(output_filepath, 'a') as f_output:
        for res in res_list:
            f_output.write(str(res) + '\n')


def analyze(train_y_list, obsv_rating_num, five_star_um_dic, one_star_um_dic):
    T_train_num = len(train_y_list)
    T_pos_neg_ratio = float(sum(1 for x in train_y_list if x > 0)) / float(sum(1 for x in train_y_list if x < 0))
    T_train_1234 = len(five_star_um_dic[1234]) * len(one_star_um_dic[1234]) * 2
    T_train_4321 = len(five_star_um_dic[4321]) * len(one_star_um_dic[4321]) * 2

    print "\n# ----- # TRAINING SET STATISTICS * START # ----- #"
    print "Total number of observed ratings in R: " + str(obsv_rating_num)
    print "Total number of training examples in T: " + str(T_train_num)
    print "Ratio of positive examples to negative examples in T: " + str(T_pos_neg_ratio)
    print "Number of training examples in T for user ID 1234: " + str(T_train_1234)
    print "Number of training examples in T for user ID 4321: " + str(T_train_4321)
    print "# ----- # TRAINING SET STATISTICS * END # ----- #\n"