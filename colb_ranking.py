__author__ = 'luoshalin'

import numpy as np
from sklearn.svm import SVC


def get_colbrk_train(R, U, V, five_star_um_dic, one_star_um_dic):  # R: sparse matrix  U,V: np matrix
    user_size = R.shape[0]
    movie_size = R.shape[1]

    v_list = []  # vector list
    y_list = []  # label list

    for u in five_star_um_dic.keys():
        if u not in one_star_um_dic:
            continue
        print "user#" + str(u)
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

                # save <v_1, y_1> and <v_2, y_2> into training matrix T
                v_list.append(v_1)
                v_list.append(v_2)
                y_list.append(y_1)
                y_list.append(y_2)
    return v_list, y_list


def get_colbrk_pred(uid_list, mid_list, U, V):
    pred_v_list = []
    for uid, mid in zip(uid_list, mid_list):
        v = np.multiply(U[uid], V[mid])
        pred_v_list.append(v)
    return pred_v_list


def svm(train_v_list, train_y_list, pred_v_list):  # T is the input matrix
    # train
    print "SVM model: begin training..."
    svm_model = SVC()
    svm_model.fit(train_v_list, train_y_list)
    print "### SVM model: end training! ###\n"
    # predict
    print "Begin predicting SVM model..."
    pred_res_list = svm_model.predict(pred_v_list)
    print "### SVM model: end predicting! ###\n"
    return pred_res_list


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