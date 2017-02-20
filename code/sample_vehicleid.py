import scipy.io
import random
import util
import numpy as np
import pickle


def sample_data(src_file, dst_file, batch_cnt):
    file_w = open(dst_file, 'w')
    lines = [line.rstrip('\n') for line in open(src_file)]
    dict_small_class = []
    dict_large_class = []
    for i in range(5043):
        dict_small_class.append([])
    for i in range(223):
        dict_large_class.append([])
    for i in xrange(len(lines)):
        cur_str = lines[i]
        split_str = cur_str.split(' ')
        if (int(split_str[2])) not in dict_large_class[int(split_str[1])]:
            dict_large_class[int(split_str[1])].append(int(split_str[2]))
        dict_small_class[int(split_str[2])].append(split_str[0])

    for i in range(batch_cnt):
        large_class_list = random.sample(range(223), 2) #sample 2 big classes
        for j in large_class_list:
            if (len(dict_large_class[j]) > 10) : #sample 10 small classes for each sub-class
                small_class_list = random.sample(dict_large_class[j], 10)
            else:
                small_class_list = dict_large_class[j]
            for k in small_class_list:
                if (len(dict_small_class[k]) > 5):
                    image_list = random.sample(dict_small_class[k], 5)
                else:
                    image_list = dict_small_class[k]
                for image_ in image_list:
                    cur_line = "".join(image_) + "\t" + str(k) + "\n"
                    file_w.write(cur_line)
    file_w.close()

def shuffle_data(mode):
    print 'shuffle product dataset ...'
    if mode == 'test':
        shuffle_file = open('../../Data/StanfordProducts/stanford_products_test_shuffle.txt', 'w')
        lines_file = [line.rstrip('\n') for line in open('../../Data/StanfordProducts/stanford_products_test.txt')]
    else:
        shuffle_file = open('../../Data/StanfordProducts/stanford_products_train_shuffle.txt', 'w')
        lines_file = [line.rstrip('\n') for line in open('../../Data/StanfordProducts/stanford_products_train.txt')]
    num_lines = len(lines_file)
    random_nums = random.sample(xrange(num_lines), num_lines)
    for i in random_nums[:num_lines]:
        shuffle_file.write(lines_file[i] + "\n")
    shuffle_file.close()

def shuffle_pair_sample_data():
    pos_pairs, neg_pairs = pickle.load(open('../../Data/StanfordProducts/pos_neg_pairs.dat', 'rb'))
    pos_pair_size = len(pos_pairs)
    neg_pair_size = len(neg_pairs)
    all_pair_size = pos_pair_size + neg_pair_size
    all_pair_list = pos_pairs + neg_pairs
    random_nums = random.sample(xrange(all_pair_size), all_pair_size)
    shuffle_pair_list_temp = []
    shuffle_pair_list = []
    for i in xrange(all_pair_size):
        shuffle_pair_list_temp.append(all_pair_list[random_nums[i]])
    random_nums = random.sample(xrange(all_pair_size), all_pair_size)
    for i in xrange(all_pair_size):
        shuffle_pair_list.append(shuffle_pair_list_temp[random_nums[i]])
    pickle.dump(shuffle_pair_list, open('../../Data/StanfordProducts/shuffle_pairs_dataset.dat', 'wb'), True)

def write_shuffle_pair_data():
    file_w = open('../../Data/StanfordProducts/stanford_products_all_pair_train.txt', 'w')
    shuffle_pair_list = pickle.load(open('../../Data/StanfordProducts/shuffle_pairs_dataset.dat', 'rb'))
    shuffle_pair_size = len(shuffle_pair_list)
    for i in xrange(shuffle_pair_size) :
        line_1 = "".join(shuffle_pair_list[i][0]) + ".jpg"+ "\t" + str(shuffle_pair_list[i][2]) + "\n"
        line_2 =  "".join(shuffle_pair_list[i][1]) + ".jpg" + "\t" + str(shuffle_pair_list[i][3]) + "\n"
        file_w.write(line_1)
        file_w.write(line_2)
    file_w.close()

def write_test_data(choice):
    root_path = "D:\\users\\v-yuhyua\\fromGPU02\\data\\VehicleID\\images\\"
    if choice == 'train':
        file_w = open('../../Data/VehicleID/VehicleID_train.txt', 'w')
        lines = [line.rstrip('\n') for line in open('../../Data/VehicleID/train_3cols.txt')]
    else:
        file_w = open('../../Data/VehicleID/val_c10.txt', 'w')
        lines = [line.rstrip('\n') for line in open('../../Data/VehicleID/val.txt')]
    for i in xrange(len(lines)):
        cur_str = lines[i]
        split_str = cur_str.split(' ')
        cur_line = root_path + split_str[0] + "\t" + split_str[2] + "\n"
        file_w.write(cur_line)
    file_w.close()

def tab2space():
    file_w = open('D://users//v-yuhyua//fromGPU02//test_model//Vehicle_ID//t2s_test_800.txt', 'w')
    lines = [line.rstrip('\n') for line in open('D://users//v-yuhyua//fromGPU02//test_model//Vehicle_ID//test_list_800.txt')]
    for i in xrange(len(lines)):
        cur_str = lines[i]
        split_str = cur_str.split(' ')
        cur_line = split_str[0] + ".jpg " + split_str[1] + "\n"
        file_w.write(cur_line)
    file_w.close()
