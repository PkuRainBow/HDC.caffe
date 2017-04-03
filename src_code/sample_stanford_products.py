import random
import numpy as np
import util

# the missing_images are the images that we can not find in the dataset.
missing_images = ['lamp_final/221777566426_7', 'bicycle_final/231588927256_1', 'fan_final/131546166499_5',
                  'coffee_maker_final/291488772961_1', 'fan_final/280961317122_7', 'lamp_final/271802113987_2']

def create_training_data(src_file, dst_file, class_cnt, batch_cnt):
    """
    param:
        src_file / dst_file : the source txt files
        class_cnt: the number of classes in the files
        sample_cnt: the number of the training mini-batch we sample
    return :
        the sampled files following the 2 level stategy, firstly choose 2 big classes, next choose 10 small classes for every big class
    """
    print 'sample training data ...'
    lines = [line.rstrip('\n') for line in open(src_file)]
    file_w = open(dst_file, 'w')
    bias = 1 # the label start from 1, so we need to subtract it by 1
    dict_small_class = []
    dict_large_class = []
    prob_small_class = []
    cnt_large_class = np.zeros(12)
    cnt_small_class = np.zeros(class_cnt)
    prob_large_class = np.zeros(12)
    img_cnt = 0
    for i in range(class_cnt):
        dict_small_class.append([])
    for i in range(12):
        dict_large_class.append([])
        prob_small_class.append([])
    for i in xrange(len(lines)):
        if i == 0 :
            continue
        cur_str = lines[i]
        split_str = cur_str.split(' ')
        if (int(split_str[1]) - bias) not in dict_large_class[int(split_str[2]) - 1]:
            dict_large_class[int(split_str[2]) - 1].append(int(split_str[1]) - bias)
        cur_name_str = split_str[3].split('.')
        if cur_name_str[0] in missing_images :
            continue
        dict_small_class[int(split_str[1]) - bias].append(cur_name_str[0])
        cnt_small_class[int(split_str[1]) - bias] += 1
        cnt_large_class[int(split_str[2]) - 1] += 1
        img_cnt += 1
    #get data distribution
    for i in range(12):
        prob_large_class[i] = cnt_large_class[i]/img_cnt
        for j in dict_large_class[i]:
            prob_small_class[i].append(cnt_small_class[j]/[cnt_large_class[i]])
    for i in range(batch_cnt):
        # step 1 : sample 2 big classes according to the probabilities
        large_class_list = np.random.choice(range(12), 2, replace=False, p=prob_large_class)
        for j in large_class_list:
            # step 2 : sample 10 different small classes in the 2 seleted big classes
            set_int = np.asarray(dict_large_class[j]).reshape(len(dict_large_class[j]), 1)
            set_prob = np.asarray(prob_small_class[j]).reshape(len(prob_small_class[j]), 1)
            image_list = np.random.choice(set_int[:,0], 10, replace=False, p=set_prob[:,0])
            for k in image_list:
                # step 3 : sample all the images of each sampled sub-class
                for m in dict_small_class[k]:
                    # cur_line = "".join(m) + ".jpg" + "\t" + str(k) + "\n"
                    cur_line = "".join(m) + ".jpg" + " " + str(k) + "\n"
                    file_w.write(cur_line)
    file_w.close()

def preprocess_data(src_file, dst_file):
    print 'preprocess training data ...'
    file_w = open(dst_file, 'w')
    lines = [line.rstrip('\n') for line in open(src_file)]
    for i in xrange(len(lines)):
        if i == 0 :
            continue
        cur_str = lines[i]
        split_str = cur_str.split(' ')
        cur_name_str = split_str[3].split('.')
        if cur_name_str[0] in missing_images :
            continue
        cur_line = "".join(split_str[3]) + " " + split_str[1] + "\n"
        file_w.write(cur_line)
    file_w.close()


def create_dat(src_file, label_file):
    util.dump_label(src_file, label_file, 1, '\t')

# sample_train_dst_file = '../data/stanford_products/train_cvpr16.txt'
# sample_pair_method(train_src_file, sample_train_dst_file, 11318)
#
# create_dat(test_file, test_label_dat)


