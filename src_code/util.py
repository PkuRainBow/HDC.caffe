"""
__file__

    utils.py

__description__

    This file provides functions for
        1. norm_1d

__author__

    RainbowSecret <yhyuan@pku.edu.cn>

"""
import random
import lmdb
import caffe
import os
import shutil
import errno
import numpy as np
import cPickle as pickle
from caffe.proto import caffe_pb2
import matplotlib.pyplot as plt

def sign_power(v):
    rows,cols=v.shape
    for i in range(rows):
        cur_sign = np.sign(v[i,:])
        v[i,:] = abs_normalize_1d(v[i, :])
        v[i,:] = np.dot(v[i,:], cur_sign)
    return v

def abs_normalize_1d(v) :
    norm = np.linalg.norm(v)
    v = np.abs(v)
    if norm==0:
        return v
    return v/norm

def normalize_1d(v) :
    norm = np.linalg.norm(v)
    if norm==0:
        return v
    return v/norm

def sign_power(v):
    abs_v = np.abs(v)
    power_v = np.power(abs_v, 0.5)
    sign_v = np.sign(v);
    return np.multiply(sign_v, power_v)

def binarize(v, feature_dim):
    one_hot = np.zeros(feature_dim)
    one_hot[np.argmax(v)] = 1
    return one_hot

def binarize_matrix_by_row(matrix):
    row, col = matrix.shape
    binary_matrix = np.zeros((row, col))
    for i in range(row):
        binary_matrix[np.argmax(matrix[i, :])] = 1
    return binary_matrix

def normalize_2d(v):
    rows,cols=v.shape
    for i in range(rows):
        v[i,:] = normalize_1d(v[i,:])
    return v

def shuffle_split(data_txt, train_lines, shuffle_train_txt, shuffle_test_txt):
    """Split the txt File, lines[:train_lines] will be split as training data, the rest as test data
    """
    print 'shuffle split dataset ...'
    shuffle_train_writer = open(shuffle_train_txt, 'w')
    shuffle_test_writer = open(shuffle_test_txt, 'w')
    lines = [line.rstrip('\n') for line in open(data_txt)]
    num_lines = len(lines)
    random_nums = random.sample(xrange(num_lines), num_lines)
    for i in random_nums[:train_lines]:
        shuffle_train_writer.write(lines[i] + "\n")
    shuffle_train_writer.close()
    for i in random_nums[train_lines + 1:]:
        shuffle_test_writer.write(lines[i] + "\n")
    shuffle_test_writer.close()

def dump_label(data_txt, dump_file, col_index, split_char):
    """change the txt file to dump store
    """
    print 'dump label data ...'
    lines = [line.rstrip('\n') for line in open(data_txt)]
    num_lines = len(lines)
    # data_vector = np.zeros(num_lines, dtype=np.int16)
    data_vector = ["" for x in range(num_lines)]
    for i in xrange(len(lines)):
        # print 'data_txt ' + str(i)
        line = lines[i]
        split_line = line.split(split_char)
        if len(split_line) < 2:
            data_vector[i] = i
        else:
            data_vector[i] = split_line[col_index]
    pickle.dump(data_vector, open(dump_file, 'wb'), True)

def create_dat(src_file, label_file):
    print 'create dat file ...'
    dump_label(src_file, label_file, 1, '\t')

def load_label(dump_file):
    label_vector = pickle.load(open(dump_file, 'rb'))
    return label_vector

def load_lmdb(lmdb_path, feature_cnt, feature_dim, pre_mode):
    """change the txt file to dump store

       Args:
            pre_mode : [0 no process] [1 : L2-normalize] [2 : binarize]
       Returns:
            data_feature_vector
    """
    print 'loading lmdb ...'
    feature_lmdb_env = lmdb.open(lmdb_path)
    lmdb_txn = feature_lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()
    num = feature_cnt
    data_feature_vector = np.zeros((num, feature_dim), dtype=np.float64)
    for ix,(key, value) in enumerate(lmdb_cursor):
        if ix == feature_cnt:
            break
        # datum.ParseFromString(key)
        # data = caffe.io.datum_to_array(datum)
        # image_name = data[0]
        datum.ParseFromString(value)
        data = caffe.io.datum_to_array(datum)
        data = np.squeeze(data)[:]
        data_feature_vector[ix, :] = data
        if pre_mode > 0:
            if pre_mode == 1:
                data_feature_vector[ix, :] = normalize_1d(data)
            elif pre_mode == 3:
                cache = sign_power(data)
                data_feature_vector[ix, :] = cache
            else:
                data_feature_vector[ix, :] = binarize(data, feature_dim)
        if (ix+1)%2000==0:
            print 'feature process %d' %(ix+1)
    print 'finished loading lmdb ...'
    return  data_feature_vector

def load_lmdb_label(lmdb_path, label_cnt):
    """load the label vector

       Returns:
            label_vector
    """
    print 'loading lmdb label vector ...'
    feature_lmdb_env = lmdb.open(lmdb_path)
    lmdb_txn = feature_lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()
    label_vector = np.zeros((label_cnt), dtype=np.int32)
    for ix,(key, value) in enumerate(lmdb_cursor):
        if ix == label_cnt:
            break
        datum.ParseFromString(value)
        label_vector[ix] = datum.label
        data = caffe.io.datum_to_array(datum)
        data = np.squeeze(data)[:]
        if (ix+1)%2000==0:
            print 'label vector process %d' %(ix+1)
    print 'finished loading lmdb label ...'
    return  label_vector

def copy_folder(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc:
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else:
            raise

def create_dirs(cur_path, class_cnt):
    for i in xrange(class_cnt):
        class_path = cur_path + str(i) + '\\'
        if not os.path.exists(class_path):
            os.makedirs(class_path)

def plot_hiotogram(dists, test_label_vector, data_label_vector, plt_name, y_max):
    num_query = dists.shape[0]
    num_data = dists.shape[1]
    dist_pos = []
    dist_neg = []
    for i in xrange(num_query):
        for j in xrange(num_data):
            if i == j: continue
            if test_label_vector[i] == data_label_vector[j]:
                dist_pos.append(dists[i,j])
            else:
                dist_neg.append(dists[i,j])
    pos_mean = np.mean(dist_pos)
    pos_var = np.var(dist_pos)
    neg_mean = np.mean(dist_neg)
    neg_var = np.var(dist_neg)
    print "pos pairs mean : ", pos_mean
    print "pos pairs variance : ", pos_var
    print "neg pairs mean : ", neg_mean
    print "neg pairs variance : ", neg_var
    print "fisher number: ", ((neg_mean-pos_mean) * (neg_mean-pos_mean))/(pos_var + neg_var)
    n, bins, patches = plt.hist(dist_pos, 200, normed=1, facecolor='green', alpha=0.5)
    n, bins, patches = plt.hist(dist_neg, 200, normed=1, facecolor='red', alpha=0.5)
    plt.ylim((0, y_max))
    plt.subplots_adjust(left=0.15)
    plt.title(plt_name)
    plt.show()