import scipy.io
import random
import util
import numpy as np
import pickle

def dump_label(data_txt, dump_file, col_index, split_char):
    """change the txt file to dump store

    """
    print 'dump label data ...'
    lines = [line.rstrip('\n') for line in open(data_txt)]
    num_lines = len(lines)
    # data_vector = np.zeros(num_lines, dtype=np.int16)
    data_vector = ["" for x in range(num_lines)]
    for i in xrange(len(lines)):
        print 'data_txt ' + str(i)
        line = lines[i]
        split_line = line.split(split_char)
        if len(split_line) < 2:
            data_vector[i] = i
        else:
            # data_vector[i] = int(split_line[col_index])
            data_vector[i] = split_line[col_index]
    pickle.dump(data_vector, open(dump_file, 'wb'), True)

def load_label(dump_file):
    label_vector = pickle.load(open(dump_file, 'rb'))
    return label_vector

def sample_mnist_data(mode):
    if mode == 'train':
        lines = [line.rstrip('\n') for line in open('D:/users/v-yuhyua/fromGPU02/data/mnist/train.txt')]
        file_w = open('D:/users/v-yuhyua/fromGPU02/data/mnist/train_.txt', 'w')
    else :
        lines = [line.rstrip('\n') for line in open('D:/users/v-yuhyua/fromGPU02/data/mnist/test.txt')]
        file_w = open('D:/users/v-yuhyua/fromGPU02/data/mnist/test_.txt', 'w')

    dict_list = []
    for i in range(10):
        dict_list.append([])

    for i in xrange(len(lines)):
        print i
        cur_str = lines[i]
        split_str = cur_str.split(' ')
        dict_list[int(split_str[1])].append(split_str[0])
        # if dict_list[int(split_str[1])].length() <= 50:
        #     cur_line = "mnist_" + "".join(split_str[0]) + " " + split_str[1] + "\n"
        #     file_w.write(cur_line)
        cur_line = "mnist_" + "".join(split_str[0]) + " " + split_str[1] + "\n"
        file_w.write(cur_line)
    file_w.close()

    # for i in range(1):
    #     print i
    #     for j in range(10):
    #         cur_size = len(dict_list[j])
    #         image_list = random.sample(range(cur_size), 100)
    #         for k in image_list:
    #             cur_line = "mnist_" + "".join(dict_list[j][k]) + " " + str(j) + "\n"
    #             file_w.write(cur_line)
    # file_w.close()

# sample_mnist_data('train')

data_txt = 'D:/users/v-yuhyua/fromGPU02/data/mnist/test.txt'
dump_file = 'D:/users/v-yuhyua/fromGPU02/data/mnist/test_label.dat'
dump_label(data_txt, dump_file, 1, ' ')
test_vec = load_label(dump_file)
print "test"
