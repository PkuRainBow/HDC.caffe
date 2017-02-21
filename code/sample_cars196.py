import random
import scipy.io as sio
from PIL import Image

def preprocess_data(src_file, dst_file, choice):
    print choice, 'preprocess ...'
    if choice == 'train':
        file_w = open(dst_file, 'w')
        lines = [line.rstrip('\n') for line in open(src_file)]
    else:
        file_w = open(dst_file, 'w')
        lines = [line.rstrip('\n') for line in open(src_file)]
    if choice == 'train':
        for i in range(0, 8054):
            cur_str = lines[i]
            split_str = cur_str.split(' ')
            image_str = (split_str[0]).split('/')
            cur_line = image_str[1] + " " + split_str[1] + "\n"
            file_w.write(cur_line)
        file_w.close()
    else:
        for i in range(8054, 16185):
            cur_str = lines[i]
            split_str = cur_str.split(' ')
            image_str = (split_str[0]).split('/')
            cur_line = image_str[1] + " " + split_str[1] + "\n"
            file_w.write(cur_line)
        file_w.close()

def sample_data(src_file, dst_file, class_cnt, batch_cnt):
    print 'sample training data ...'
    file_w = open(dst_file, 'w')
    lines = [line.rstrip('\n') for line in open(src_file)]
    dict_small_class = []
    for i in range(98):
        dict_small_class.append([])
    for i in xrange(len(lines)):
        cur_str = lines[i]
        split_str = cur_str.split(' ')
        dict_small_class[int(split_str[1])].append(split_str[0])
    for i in range(batch_cnt):
        small_class_list = random.sample(range(98), class_cnt)
        for j in small_class_list:
            image_list = random.sample(dict_small_class[j], 100/class_cnt)
            for k in image_list:
                # cur_line = k + "\t" + str(j) + "\n"
                cur_line = k + " " + str(j) + "\n"
                file_w.write(cur_line)
    file_w.close()

def crop_images(src_file, input_dir, output_dir, crop_file):
    cars_ = sio.loadmat(crop_file)
    anno_ = cars_['annotations']
    lines_image = [line.rstrip('\n') for line in open(src_file)]
    for i in xrange(16185):
        line = lines_image[i]
        elem = line.split(' ')
        x_1 = int(anno_[0][i][1])
        y_1 = int(anno_[0][i][2])
        x_2 = int(anno_[0][i][3])
        y_2 = int(anno_[0][i][4])
        dir_split_ = elem[0].split('/')
        image_ = Image.open(input_dir + str(elem[0]))
        crop_image = image_.crop((x_1, y_1, x_2, y_2))
        crop_image.save(output_dir + dir_split_[1])
        print i