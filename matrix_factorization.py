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


def pmf(R):
    # -------/ INITIALIZATION /------- #
    # PARAMS
    d = 20  # the feature number dimension of U & V (n_factor)
    lmd = 0.01
    step = 0.0001
    itr = 200
    threshold = 2000  # stopping criteria

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
    U = np.random.rand(user_size, d)
    V = np.random.rand(movie_size, d)

    # -------/ FACTORIZATION /------- #
    last_error = 0.0
    for i in range(itr):
        A = -np.multiply(I, (R - U.dot(V.T)))
        U = U - step * A.dot(V) + step * lmd * U
        A = -np.multiply(I, (R - U.dot(V.T)))
        V = V - step * A.T.dot(U) + step * lmd * V

        new_error = cal_error(R, U, V, I)
        if abs(new_error - last_error) < threshold:
            break
        last_error = cal_error(R, U, V, I)

        print "Iteration:%d,E:%d"%(i, last_error)
    # ASSERT: GOT U & V
    # print self._U.shape
    # print self._V.shape


def cal_error(R, U, V, I):
    # print 'error:'
    # print R.shape
    # print np.dot(U, V.T).shape
    # print I.shape
    tmpM1 = np.dot(U, V.T)
    tmpM2 = R - tmpM1
    # I = sparse.csr_matrix(I)
    # print type(I)
    # print type(tmpM2)
    # tmpM3 = I * tmpM2
    tmpM3 = I * np.asarray(tmpM2, dtype=np.float64)
    tmpM4 = tmpM3 ** 2
    error = np.sum(tmpM4)
    # error = np.sum((I * (R - np.dot(U, V.T))) ** 2)
    return error