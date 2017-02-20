import scipy.io
import random
import numpy
import pickle
import os
from PIL import Image

def preprocess_data(src_file, dst_file, choice):
    print choice, 'preprocess ...'
    file_w = open(dst_file, 'w')
    lines = [line.rstrip('\n') for line in open(src_file)]
    if choice == 'train':
        for i in range(0, 5864):
            cur_str = lines[i]
            split_str = cur_str.split(' ')
            image_str = split_str[1]
            image_label = int(image_str.split('.', 1)[0]) - 1
            cur_line = image_str + "\t" + str(image_label) + "\n"
            file_w.write(cur_line)
        file_w.close()
    else:
        for i in range(5864, 11788):
            cur_str = lines[i]
            split_str = cur_str.split(' ')
            image_str = split_str[1]
            image_label = int(image_str.split('.', 1)[0]) - 1
            #cur_line = image_str + "\t" + str(image_label) + "\n"
            cur_line = image_str + "\t" + str(image_label) + "\n"
            file_w.write(cur_line)
        file_w.close()

def sample_data(src_file, dst_file, class_cnt, batch_cnt):
    print 'sample training data ...'
    file_w = open(dst_file, 'w')
    lines = [line.rstrip('\n') for line in open(src_file)]
    dict_small_class = []
    for i in range(100):
        dict_small_class.append([])
    for i in xrange(len(lines)):
        cur_str = lines[i]
        split_str = cur_str.split('\t')
        dict_small_class[int(split_str[1])].append(split_str[0])
    for i in range(batch_cnt):
        small_class_list = random.sample(range(100), class_cnt)
        for j in small_class_list:
            image_list = random.sample(dict_small_class[j], 100/class_cnt)
            for k in image_list:
                # cur_line = k + "\t" + str(j) + "\n"
                cur_line = k + " " + str(j) + "\n"
                file_w.write(cur_line)
    file_w.close()

def box_crop(src_file, input_dir, output_dir, crop_file):
    lines_box = [line.rstrip('\n') for line in open(crop_file)]
    lines_image = [line.rstrip('\n') for line in open(src_file)]
    for i in range(11787,11789):
        line_box = lines_box[i]
        line_image = lines_image[i]
        elems_box = line_box.split()
        id = elems_box[0]
        x_1 = int(float(elems_box[1]))
        y_1 = int(float(elems_box[2]))
        x_2 = int(float(elems_box[3]))
        y_2 = int(float(elems_box[4]))
        elems_image = line_image.split()
        image_name = elems_image[1]
        image_ = Image.open(input_dir + image_name)
        crop_image = image_.crop((x_1, y_1, x_1+x_2, y_1+y_2))
        dir_split_ = image_name.split('/')
        sub_dir_ = output_dir + dir_split_[0]
        if not os.path.exists(sub_dir_):
            os.makedirs(sub_dir_)
        crop_image.save(output_dir + image_name)
        print i
