from util import *
from eval_metric import *

def get_recall(dists, test_label_vector, data_label_vector, top_count):
    num_query = dists.shape[0]
    correct_radio = np.zeros((6, num_query))
    for i in xrange(num_query):
        #print label_vector[i]
        labels_sorted = data_label_vector[np.argsort(dists[i,:])].flatten()
        labels = labels_sorted[0:top_count]
        correct_count = np.zeros((6, 1))
        for j in xrange(top_count):
            if labels[j] == test_label_vector[i]:
                if (j < 1) :
                    correct_count[0] = correct_count[0] + 1
                if (j < 10) :
                    correct_count[1] = correct_count[1] + 1
                if (j < 20) :
                    correct_count[2] = correct_count[2] + 1
                if (j < 30) :
                    correct_count[3] = correct_count[3] + 1
                if (j < 40) :
                    correct_count[4] = correct_count[4] + 1
                if (j < 50) :
                    correct_count[5] = correct_count[5] + 1
        correct_radio[0][i] = (correct_count[0] > 0)
        correct_radio[1][i] = (correct_count[1] > 0)
        correct_radio[2][i] = (correct_count[2] > 0)
        correct_radio[3][i] = (correct_count[3] > 0)
        correct_radio[4][i] = (correct_count[4] > 0)
        correct_radio[5][i] = (correct_count[5] > 0)
    ave_precision = np.mean(correct_radio, axis = 1)

    print 'deep fashion in_shop mean recall@ 1 : %f' % (ave_precision[0])
    print 'deep fashion in_shop mean recall@ 10 : %f' % (ave_precision[1])
    print 'deep fashion in_shop mean recall@ 20 : %f' % (ave_precision[2])
    print 'deep fashion in_shop mean recall@ 30 : %f' % (ave_precision[3])
    print 'deep fashion in_shop mean recall@ 40 : %f' % (ave_precision[4])
    print 'deep fashion in_shop mean recall@ 50 : %f' % (ave_precision[5])

def eval(query_lmdb_path, query_label_file, data_lmdb_path, data_label_file, query_cnt, data_cnt, feature_dim, top_cnt):
    pre_mode = 1  # 0 : no process 1 : L2 norm 2 : binarize
    query_label_vector = pickle.load(open(query_label_file, 'rb'))
    data_label_vector = pickle.load(open(data_label_file, 'rb'))
    data_feature_vector = load_lmdb(data_lmdb_path, data_cnt, feature_dim, pre_mode)
    label_data_vector = pickle.load(open(data_label_file, 'rb'))
    label_query_vector = pickle.load(open(query_label_file, 'rb'))
    query_feature_vector = load_lmdb(query_lmdb_path, query_cnt, feature_dim, pre_mode)
    dists = compute_distances(query_feature_vector, data_feature_vector)
    np.fill_diagonal(dists, float("inf"))
    get_recall(dists, query_label_vector, data_label_vector, top_cnt)