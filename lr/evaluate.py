__author__ = 'luoshalin'
# FILE DESCRIPTION: functions for evaluation and generate result files

import math


# write predict results (hard & soft) into a file
def print_pred_res(pred_list_hard, pred_list_soft, output_fpath):
    with open(output_fpath, 'w') as f:
        for (a, b) in zip(pred_list_hard, pred_list_soft):
            f.write(str(a) + ' ' + str(b) + '\n')


# calculate accuracy for hard result
def cal_accuracy(list1, list2):
    cnt = 0
    length = 0
    for i, j in zip(list1, list2):
        length += 1
        if i == j:
            cnt += 1

    return float(cnt) / length


# calculate rmse for soft result
def cal_rmse(list1, list2):
    n = len(list1)
    sum = 0.0
    for i in range(n):
        diff = list1[i] - list2[i]
        sum += diff * diff

    rmse = math.sqrt(sum / n)
    return rmse