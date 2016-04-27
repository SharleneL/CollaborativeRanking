__author__ = 'luoshalin'
# FILE DESCRIPTION: functions to extract features

from scipy.sparse import *


# function to get feature_set and reviews represented by freq dicts
def gen_libsvm_file(review_list, stars_list, test_review_list, method, feature_num):
    feature_str_index_dic = dict()  # dic saves top-2000 most frequent tokens <feature_str, feature_index>
    review_token_freq_dic_list = []  # list of dicts, each dicts is for one review, containing <token, freq>
    test_review_token_freq_dic_list = []  # for test data: list of dicts, each dicts is for one review, containing <token, freq>
    feature_dic = dict()  # dict saves all original features <token, corpus_ctf_cnt / corpus_df_cnt>

    if method == 'ctf':
        # training data
        for review in review_list:
            tokens = getattr(review, 'tokens')
            review_token_freq_dic = dict()
            for t in tokens:
                # update feature dic
                if t not in feature_dic:
                    feature_dic[t] = 1
                else:
                    feature_dic[t] += 1
                # update review dic
                if t not in review_token_freq_dic:
                    review_token_freq_dic[t] = 1
                else:
                    review_token_freq_dic[t] += 1
            review_token_freq_dic_list.append(review_token_freq_dic)
        # testing data
        for review in test_review_list:
            tokens = getattr(review, 'tokens')
            review_token_freq_dic = dict()
            for t in tokens:
                # update review dic
                if t not in review_token_freq_dic:
                    review_token_freq_dic[t] = 1
                else:
                    review_token_freq_dic[t] += 1
            test_review_token_freq_dic_list.append(review_token_freq_dic)

    elif method == 'df':
        # training data
        for review in review_list:
            tokens = getattr(review, 'tokens')
            tokens_set = set(tokens)  # use set here, to judge whether a token appears in one review

            # update feature dic
            for t in tokens_set:
                if t not in feature_dic:
                    feature_dic[t] = 1
                else:
                    feature_dic[t] += 1

            # update review dic
            review_token_freq_dic = dict()
            for t in tokens:
                if t not in review_token_freq_dic:
                    review_token_freq_dic[t] = 1
                else:
                    review_token_freq_dic[t] += 1
            review_token_freq_dic_list.append(review_token_freq_dic)

        # testing data
        for review in test_review_list:
            tokens = getattr(review, 'tokens')
            # update review dic
            review_token_freq_dic = dict()
            for t in tokens:
                if t not in review_token_freq_dic:
                    review_token_freq_dic[t] = 1
                else:
                    review_token_freq_dic[t] += 1
            test_review_token_freq_dic_list.append(review_token_freq_dic)

    else:
        print 'ERROR: Invalid feature extraction method: ' + method

    sorted_feature_list = sorted(feature_dic.items(), key=lambda x: -x[1])

    for i in range(feature_num):  # 10 for testing, or 2000 for experiment
        feature_str_index_dic[sorted_feature_list[i][0]] = i

    # output to a dict file
    with open(method+'_top_features.list', 'w') as feature_f:
        for i in range(feature_num):  # 10 for testing, or 2000 for experiment
            feature_f.write(str(sorted_feature_list[i][0]) + '\t' + str(sorted_feature_list[i][1]) + '\n')

    train_output_fpath = method + '_libsvm_train'+'.out'
    test_output_fpath = method + '_libsvm_test'+'.out'

    # training data
    with open(train_output_fpath, 'w') as output_f:
        for i in range(len(review_token_freq_dic_list)):  # each elem d in the list is a dict for one review
            output_str = ''
            d = review_token_freq_dic_list[i]

            for token, freq in d.iteritems():
                if token in feature_str_index_dic:
                    output_str = output_str + str(feature_str_index_dic[token]) + ':' + str(d[token]) + ' '
            output_str = str(stars_list[i]) + ' ' + output_str.strip() + '\n'
            output_f.write(output_str)

    # testing data
    with open(test_output_fpath, 'w') as output_f:
        for i in range(len(test_review_token_freq_dic_list)):  # each elem d in the list is a dict for one review
            output_str = ''
            d = test_review_token_freq_dic_list[i]

            for token, freq in d.iteritems():
                if token in feature_str_index_dic:
                    output_str = output_str + str(feature_str_index_dic[token]) + ':' + str(d[token]) + ' '
            output_str = output_str.strip() + '\n'
            output_f.write(output_str)

    return train_output_fpath, test_output_fpath


# get the train & eval feature matrix: row - one review, col - token's index in feature dic, value - occurence frequency
def get_train_feature_M(libsvm_file_path, total_data_amt, eval_data_frac, feature_num, class_num):

    # train SET
    # stars
    train_stars_list = []     # list of stars for all reviews
    train_stars_row = []      # review data index (AKA data's line number)
    train_stars_col = []      # class 1~5
    train_stars_data = []     # 1: current review's star = col+1  0: current review's star is not related to this col
    # review features
    train_review_row = []     # review data index (AKA data's line number)
    train_review_col = []     # feature's index
    train_review_data = []    # the frequency of <review_data_index, feature_index>

    # eval SET
    # stars
    eval_stars_list = []     # list of stars for all reviews
    eval_stars_row = []      # review data index (AKA data's line number)
    eval_stars_col = []      # class 1~5
    eval_stars_data = []     # 1: current review's star = col+1  0: current review's star is not related to this col
    # review features
    eval_review_row = []     # review data index (AKA data's line number)
    eval_review_col = []     # feature's index
    eval_review_data = []    # the frequency of <review_data_index, feature_index>

    # split the data into train & eval parts
    train_data_amt = int(round(total_data_amt * (1.0 - eval_data_frac)))

    line_num = 0
    with open(libsvm_file_path) as f:
        line = f.readline().strip()
        while line != '':
            # split into training set
            arr = line.split()
            if line_num < train_data_amt:
                # update stars
                stars = int(arr[0])
                train_stars_list.append(stars)
                # star matrix
                train_stars_col.append(stars-1)
                train_stars_row.append(line_num)
                train_stars_data.append(1)

                # update review features
                arr = line.split()
                for feature in arr[1:]:
                    index_cnt_arr = feature.split(':')  # feature index
                    index = int(index_cnt_arr[0])
                    cnt = int(index_cnt_arr[1])

                    train_review_row.append(line_num)
                    train_review_col.append(index)
                    train_review_data.append(cnt)

            # split into evaluating set
            else:
                # update stars
                stars = int(arr[0])
                eval_stars_list.append(stars)
                # star matrix
                eval_stars_col.append(stars-1)
                eval_stars_row.append(line_num - train_data_amt)  # shall subtract train data amount from line number
                eval_stars_data.append(1)

                # update review features
                arr = line.split()
                for feature in arr[1:]:
                    index_cnt_arr = feature.split(':')  # feature index
                    index = int(index_cnt_arr[0])
                    cnt = int(index_cnt_arr[1])

                    eval_review_row.append(line_num - train_data_amt)  # shall subtract train data amount from line number
                    eval_review_col.append(index)
                    eval_review_data.append(cnt)

            line = f.readline()
            line_num += 1

    train_M = csr_matrix((train_review_data, (train_review_row, train_review_col)), shape=(train_data_amt, feature_num))
    eval_M = csr_matrix((eval_review_data, (eval_review_row, eval_review_col)), shape=(line_num - train_data_amt, feature_num))
    train_stars_M = csr_matrix((train_stars_data, (train_stars_row, train_stars_col)), shape=(train_data_amt, class_num))
    eval_stars_M = csr_matrix((eval_stars_data, (eval_stars_row, eval_stars_col)), shape=(line_num - train_data_amt, class_num))

    return train_M, eval_M, train_stars_M, eval_stars_M, eval_stars_list


# generate feautures for test/dev set
def get_test_feature_M(test_libsvm_file_path, feature_num):
    test_review_row = []
    test_review_col = []
    test_review_data = []

    line_num = 0
    with open(test_libsvm_file_path) as f:
        line = f.readline().strip()
        while line != '':
            arr = line.split()
            for feature in arr:  # start from index 0
                index_cnt_arr = feature.split(':')  # feature index
                index = int(index_cnt_arr[0])
                cnt = int(index_cnt_arr[1])

                test_review_row.append(line_num)
                test_review_col.append(index)
                test_review_data.append(cnt)

            line = f.readline()
            line_num += 1

    test_M = csr_matrix((test_review_data, (test_review_row, test_review_col)), shape=(line_num, feature_num))

    return test_M