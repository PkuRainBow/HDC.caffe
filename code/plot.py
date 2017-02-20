import numpy as np
import matplotlib.pyplot as plt

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