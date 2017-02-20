import numpy as np
from numpy import linalg as la

def compute_distances(query_list, set_list):
    """
    param: query_list is 2D array [query_cnt, feature_dim]
           set_list is 2D array [set_cnt, feature_dim]
    return: dists is 2D array [query_cnt, set_cnt], L2-distance
    """
    print 'start compute distances ...'
    num_query = query_list.shape[0]
    num_set = set_list.shape[0]
    dists = np.zeros((num_query, num_set))
    M = np.dot(query_list, set_list.T)
    te = np.square(query_list).sum(axis = 1)
    tr = np.square(set_list).sum(axis = 1)
    dists = np.sqrt(np.abs(-2*M + tr + np.matrix(te).T))
    # for i in xrange(num_query):
    #     for j in xrange(num_set):
    #         dists[i,j] = np.linalg.norm(query_list[i,:] - set_list[j,:])
    print 'finished compute distances ...'
    return dists

def compute_distances_self(set_list):
    print 'start compute distances ...'
    num_set = set_list.shape[0]
    dists = np.zeros((num_set, num_set))
    M = np.dot(set_list, set_list.T)
    te = np.square(set_list).sum(axis = 1)
    tr = np.square(set_list).sum(axis = 1)
    dists = np.sqrt(np.abs(-2*M + tr + np.matrix(te).T))
    print 'finished compute distances ...'
    return dists

def get_eval(dists, test_label_vector, data_label_vector, top_count):
    """
    param: dists is the distance matrix calculated by compuate_distance(..)
           test_label_vector [test_cnt, 1]/ data_label_vector[data_cnt, 1] is the label for the query images and dataset images
           top_count is used to determine we calculate the MAP@top_count and Precision@top_count
    """
    num_query = dists.shape[0]
    precision = np.zeros((num_query, 1))
    correct_radio = np.zeros((num_query, 1))
    for i in xrange(num_query):
        labels_sorted = data_label_vector[np.argsort(dists[i,:])].flatten()
        labels = labels_sorted[0:top_count]
        correct_count = 0
        for j in xrange(top_count):
            if labels[j] == test_label_vector[i]:
                correct_count = correct_count + 1
                precision[i] += correct_count / float(j+1)
        if correct_count > 0 :
            precision[i] = precision[i] / float(correct_count)
        correct_radio[i] = correct_count / float(top_count)
    ave_map = np.mean(precision)
    ave_precision = np.mean(correct_radio)
    return ave_map, ave_precision




