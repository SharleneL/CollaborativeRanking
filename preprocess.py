__author__ = 'luoshalin'


from scipy.sparse import *
from scipy import *


def get_trainM(filepath):
    user = []   # as row
    movie = []  # as col
    score = []  # as data

    # saves the total number of users & movies
    user_size = 0
    movie_size = 0

    # save <uid, mid_list>
    five_star_um_dic = dict()
    one_star_um_dic = dict()

    obsv_rating_num = 0
    with open(filepath) as f:
        line = f.readline().strip()
        while line != '':
            obsv_rating_num += 1
            movie_num = int(line.split(',')[0])
            user_num = int(line.split(',')[1])
            score_num = int(line.split(',')[2])

            movie.append(movie_num)
            user.append(user_num)
            score.append(score_num - 3)  # imputation option 2

            user_size = max(user_size, user_num)
            movie_size = max(movie_size, movie_num)

            # update dictionaries
            if score_num == 5:
                if user_num in five_star_um_dic:
                    five_star_um_dic[user_num].append(movie_num)
                else:
                    mid_list = [movie_num]
                    five_star_um_dic[user_num] = mid_list
            if score_num == 1:
                if user_num in one_star_um_dic:
                    one_star_um_dic[user_num].append(movie_num)
                else:
                    mid_list = [movie_num]
                    one_star_um_dic[user_num] = mid_list

            line = f.readline().strip()
    trainM = csr_matrix((score, (user, movie)), shape=(user_size+1, movie_size+1))  # because user & movie index starts from 0; total user number should be user_size+1
    return trainM, five_star_um_dic, one_star_um_dic, obsv_rating_num


# get the userlist & movielist to be predict in the dev file
def get_pred_lists(filepath):
    uid_list = []   # as row
    mid_list = []  # as col
    # score = []  # as data

    # saves the total number of users & movies
    user_size = 0
    movie_size = 0

    with open(filepath) as f:
        line = f.readline().strip()
        while line != '':
            movie_num = int(line.split(',')[0])
            user_num = int(line.split(',')[1])

            mid_list.append(movie_num)
            uid_list.append(user_num)

            user_size = max(user_size, user_num)
            movie_size = max(movie_size, movie_num)

            line = f.readline().strip()
    # trainM = csr_matrix((score, (user, movie)), shape=(user_size+1, movie_size+1))  # because user & movie index starts from 0; total user number should be user_size+1
    # return trainM, five_star_um_dic, one_star_um_dic
    return uid_list, mid_list


def get_qmM(filepath, mu_list):
    user = []   # as row
    movie = []  # as col
    score = []  # as data
    # saves the total number of users & movies
    user_size = 0
    movie_size = 0

    with open(filepath) as f:
        line = f.readline().strip()
        while line != '':
            movie_num = int(line.split(',')[0])
            user_num = int(line.split(',')[1])

            movie.append(movie_num)
            user.append(user_num)
            score.append(1)  # if value = 1, means this is a pos to be predicted
            mu_list.append((movie_num, user_num))

            user_size = max(user_size, user_num)
            movie_size = max(movie_size, movie_num)

            line = f.readline().strip()
    qmM = csr_matrix((score, (user, movie)), shape=(user_size+1, movie_size+1))  # including 0
    return qmM