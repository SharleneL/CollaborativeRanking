__author__ = 'luoshalin'

import math
import numpy as np


def gen_train(R, U, V, five_star_um_dic, one_star_um_dic):  # R: sparse matrix  U,V: np matrix
    user_size = R.shape[0]
    movie_size = R.shape[1]

    user_index_list, movie_index_list = R.nonzero()

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



    # for u in user_index_list:  # for each user, examine each movie pair
    #     print "user #" + str(u)
    #     for index_i in range(len(movie_index_list)):
    #         i = movie_index_list[index_i]
    #         for index_j in range(index_i+1, len(movie_index_list)):  # movie pair <i, j>
    #             j = movie_index_list[index_j]
    #             diff_R = R[u, i] - R[u, j]
    #             if abs(diff_R) == 4:
    #                 y = diff_R / abs(diff_R)  # y_ij
    #                 v = np.multiply(U[u], V[i]) - np.multiply(U[u], V[j])  # v_ij, is a matrix
    #                 v = np.asarray(v)[0]  # is an array
    #                 # save <v_ij, y_ij> into training matrix T
    return