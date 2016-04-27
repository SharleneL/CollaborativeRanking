__author__ = 'luoshalin'
# FILE DESCRIPTION: functions for data preprocessing

import json
import re
from classes import Review


def get_review_list(stopword_fpath, train_input_fpath, test_input_fpath):
    train_review_list = []  # list of Review obj
    test_review_list = []
    train_stars_list = []  # list of stars for all reviews
    test_stars_list = []

    f = open(stopword_fpath)
    stopword_lines = f.readlines()
    f.close()
    stopword_set = set(line.strip() for line in stopword_lines)

    print 'preprocess - line22'

    # training file
    train_line_num = 0  # i-th line
    with open(train_input_fpath) as train_input_f:
        train_line = train_input_f.readline()
        while train_line != '':
            # STEP1 - Extract useful fields
            review = json.loads(train_line)
            review_id = review['review_id']
            review_text = review['text']
            review_stars = int(review['stars'])

            # STEP2 - Tokenize the text field and remove the stop words (see stopword.list)
            # Turn all the tokens into lower case; Remove punctuation in the tokens; Remove tokens contain numbers
            review_text = review_text.lower()
            review_text = re.sub('[^A-Za-z0-9\s]+', '', review_text)
            review_token_list = review_text.split()
            tmp_review_token_list = []
            for token in review_token_list:
                if token not in stopword_set and not any(i.isdigit() for i in token):
                    tmp_review_token_list.append(token)
            review_token_list = tmp_review_token_list
            new_review = Review(review_id, review_stars, review_token_list)
            # update reviews
            train_review_list.append(new_review)
            # update stars
            train_stars_list.append(review_stars)
            train_line = train_input_f.readline()
            train_line_num += 1

    # testing file
    test_line_num = 0  # i-th line
    with open(test_input_fpath) as test_input_f:
        test_line = test_input_f.readline()
        while test_line != '':
            # STEP1 - Extract useful fields
            review = json.loads(test_line)
            review_id = review['review_id']
            review_text = review['text']
            # review_stars = int(review['stars'])

            # STEP2 - Tokenize the text field and remove the stop words (see stopword.list)
            # Turn all the tokens into lower case; Remove punctuation in the tokens; Remove tokens contain numbers
            review_text = review_text.lower()
            review_text = re.sub('[^A-Za-z0-9\s]+', '', review_text)
            review_token_list = review_text.split()
            tmp_review_token_list = []
            for token in review_token_list:
                if token not in stopword_set and not any(i.isdigit() for i in token):
                    tmp_review_token_list.append(token)
            review_token_list = tmp_review_token_list
            new_review = Review(review_id, 0, review_token_list)
            # update reviews
            test_review_list.append(new_review)
            # update stars
            # stars_list.append(review_stars)
            test_line = test_input_f.readline()
            test_line_num += 1

    return train_review_list, train_stars_list, train_line_num, test_review_list, test_stars_list, test_line_num


def print_analysis(review_list):
    # STEP3 - Calculate some statistics to verify your implementation.
    token_num_dic = dict()
    stars_list = [0, 0, 0, 0, 0, 0]
    for review in review_list:
        stars = getattr(review, 'stars')
        stars_list[stars] += 1
        tokens = getattr(review, 'tokens')
        for t in tokens:
            if t not in token_num_dic:
                token_num_dic[t] = 1
            else:
                token_num_dic[t] += 1
    tokens = sorted(token_num_dic.items(), key=lambda x: -x[1])

    # 1: top 9 most frequent tokens and corresponding counts
    print "Top 9 most frequent tokens and corresponding counts"
    for i in range(9):
        print tokens[i]

    # 2:number of training instances with 1 star, 2 stars, 3 stars, 4 stars and 5 stars
    for i in range(1, 6):
        print str(i) + " stars review count: " + str(stars_list[i])