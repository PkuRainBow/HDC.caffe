from code.eval_metric import *
from code.util import *
from code.plot import *


data_lmdb_path_cls_3 = 'D:\\users\\v-yuhyua\\fromGPU02\\feature\\StanfordProducts_FastPair_cascade_v1_hard_ratio_iter_60000_cls3'
data_lmdb_path_cls_2 = 'D:\\users\\v-yuhyua\\fromGPU02\\feature\\StanfordProducts_FastPair_cascade_v1_hard_ratio_iter_60000_cls2'
data_lmdb_path_cls_1 = 'D:\\users\\v-yuhyua\\fromGPU02\\feature\\StanfordProducts_FastPair_cascade_v1_hard_ratio_iter_60000_cls1'

# data_lmdb_path_binomial = 'D:\\users\\v-yuhyua\\fromGPU02\\feature\\StanfordProducts_binomial_deviance_iter_30000'
# data_lmdb_path_cls_3 = 'D:\\users\\v-yuhyua\\fromGPU02\\feature\\products_bvlc_feature_lmdb'
# data_lmdb_path_cls_2 = 'D:\\users\\v-yuhyua\\fromGPU02\\feature\\products_bvlc_feature_fc2_lmdb'
# data_lmdb_path_cls_1 = 'D:\\users\\v-yuhyua\\fromGPU02\\feature\\products_bvlc_feature_fc1_lmdb'

query_label_file = '../../Data/StanfordProducts/stanford_products_test_label.dat'
test_label_file = '../../Data/StanfordProducts/stanford_products_test_label.dat'

# data_feature_cnt = 60499
# query_feature_cnt = 60499
data_feature_cnt = 500

feature_dim = 128
pre_mode = 1# 0 : no process 1 : L2 norm 2 : binarize

query_label_vector = pickle.load(open(query_label_file, 'rb'))
data_label_vector = pickle.load(open(test_label_file, 'rb'))

data_feature_vector_cls_1 = load_lmdb(data_lmdb_path_cls_1, data_feature_cnt, feature_dim, pre_mode)
data_feature_vector_cls_2 = load_lmdb(data_lmdb_path_cls_2, data_feature_cnt, feature_dim, pre_mode)
data_feature_vector_cls_3 = load_lmdb(data_lmdb_path_cls_3, data_feature_cnt, feature_dim, pre_mode)

# data_feature_vector_binomial = load_lmdb(data_lmdb_path_binomial, data_feature_cnt, feature_dim, pre_mode)
data_feature_vector = np.concatenate((data_feature_vector_cls_1,   data_feature_vector_cls_2,  1 * data_feature_vector_cls_3), axis = 1)
data_feature_vector = normalize_2d(data_feature_vector)

# distance computation method
dists = compuate_distances(data_feature_vector, data_feature_vector)
# set the distances to itself to be inf
np.fill_diagonal(dists, float("inf"))
get_recall_1000(dists, query_label_vector, data_label_vector, 1010)
