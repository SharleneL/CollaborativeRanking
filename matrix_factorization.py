__author__ = 'luoshalin'

import numpy as np
from scipy import *
import sys


def pmf(R):
    # -------/ INITIALIZATION /------- #
    # PARAMS
    d = 20  # the feature number dimension of U & V (n_factor)
    lmd = 0.01
    step = 0.0001
    itr = 200
    # threshold = 2000  # stopping criteria
    # threshold = 10e-4  # stopping criteria
    threshold = 0.2  # stopping criteria

    # MATRIX
    # R: trainM
    user_size = R.shape[0]
    movie_size = R.shape[1]
    # I: I_ij=1 if useri rated moviej; =0 otherwise
    I = np.zeros(R.shape, dtype=np.float64)
    row, col = R.nonzero()  # row & col index list if R_ij != 0
    I[row, col] = 1
    # I = I.astype(np.float64, copy=False)  # TODO: do not understand why do this..
    # U:
    U = np.random.rand(user_size, d)   # (10916, 20)
    # V:
    V = np.random.rand(movie_size, d)  # (5392, 20)

    # -------/ FACTORIZATION /------- #
    last_error = sys.float_info.min
    for i in range(itr):
        A = -np.multiply(I, (R - U.dot(V.T)))
        U = U - step * A.dot(V) + step * lmd * U
        A = -np.multiply(I, (R - U.dot(V.T)))
        V = V - step * A.T.dot(U) + step * lmd * V

        new_error = cal_error(R, U, V, I)
        error_ratio = abs(float(new_error) - float(last_error)) / float(last_error)
        if error_ratio < threshold:
            break
        last_error = cal_error(R, U, V, I)

        print "Iteration #" + str(i) + "; Err changing ratio: " + str(error_ratio)
    return U, V


def cal_error(R, U, V, I):
    error = np.sum((I * np.asarray((R - np.dot(U, V.T)), dtype=np.float64)) ** 2)
    # error = np.sum((I * (R - np.dot(U, V.T))) ** 2)
    return error