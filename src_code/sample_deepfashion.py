from PIL import Image
import os
import re
import random
import util

def box_crop(input_dir, output_dir, box_file):
    # input_dir = 'D:\\users\\v-yuhyua\\fromGPU02\data\\deep_fashion\\in-shop-benchmark\\Img\\'
    # output_dir = 'D:\\users\\v-yuhyua\\fromGPU02\data\\deep_fashion\\in-shop-benchmark\\Img_bbox\\'
    # lines = [line.rstrip('\n') for line in open('../../Data/DeepFashion/in-shop/list_bbox_inshop.txt')]
    lines = [line.rstrip('\n') for line in open(box_file)]
    for i in range(0, 52712):
        line = lines[i]
        elems = line.split()
        image_name = elems[0]
        clothes_type = elems[1]
        pose_type = elems[2]
        x_1 = int(elems[3])
        y_1 = int(elems[4])
        x_2 = int(elems[5])
        y_2 = int(elems[6])
        image_ = Image.open(input_dir + image_name)
        # image_resize = image_.resize((256, 256))
        crop_image = image_.crop((x_1, y_1, x_2 - x_1, y_2 - y_1))
        # crop_image = image_.crop((x_1, y_1, x_2, y_2))
        dir_split_ = image_name.split('/')
        sub_dir_ = output_dir + dir_split_[0] + '/' + dir_split_[1] + '/' + dir_split_[2] + '/' + dir_split_[3]
        if not os.path.exists(sub_dir_):
            os.makedirs(sub_dir_)
        crop_image.save(output_dir + image_name)
        # print i

def partition_data(src_file, train_file, query_file, gallery_file):
    print 'partition data ...'
    men_clothes_type = {'Denim': 1, 'Jackets_Vests': 2, 'Pants': 3, 'Shirts_Polos': 4, 'Shorts': 5, 'Suiting': 6, 'Sweaters': 7, 'Sweatshirts_Hoodies': 8, 'Tees_Tanks': 9}
    women_clothes_type = {'Blouses_Shirts': 10, 'Cardigans': 11, 'Denim': 12, 'Dresses': 13, 'Graphic_Tees': 14, 'Jackets_Coats': 15, 'Leggings': 16, 'Pants': 17, 'Rompers_Jumpsuits': 18,
                          'Shorts': 19, 'Skirts': 20, 'Sweaters': 21, 'Sweatshirts_Hoodies': 22, 'Tees_Tanks': 23}
    lines = [line.rstrip('\n') for line in open(src_file)]
    file_train_w = open(train_file, 'w')
    file_query_w = open(query_file, 'w')
    file_gallery_w = open(gallery_file, 'w')

    train_img_id = set()
    query_img_id = set()
    gallery_img_id = set()

    for i in range(0, 52712):
        line = lines[i]
        elems = line.split()
        image_name = elems[0]
        image_id = elems[1]
        image_type = elems[2]
        class_id = re.sub("\D", "", image_id)

        big_class_id = 0
        image_elems = image_name.split('/')
        if image_elems[1] == 'MEN':
            big_class_id = men_clothes_type[image_elems[2]]
        else:
            big_class_id = women_clothes_type[image_elems[2]]
        if image_type == 'train':
            train_img_id.add(int(class_id))
            cur_line = image_name + "\t" + str(int(class_id)-1) + "\t" + str(big_class_id) + "\n"
            file_train_w.write(cur_line)
        elif image_type == 'query':
            query_img_id.add(int(class_id))
            cur_line = image_name + "\t" + str(int(class_id)-1) + "\n"
            file_query_w.write(cur_line)
        elif image_type == 'gallery':
            gallery_img_id.add(int(class_id))
            cur_line = image_name + "\t" + str(int(class_id) - 1) + "\n"
            file_gallery_w.write(cur_line)
        # print i
    file_train_w.close()
    file_query_w.close()
    file_gallery_w.close()

def sample_data(src_file, dst_file, batch_cnt):
    print 'sample data ...'
    lines = [line.rstrip('\n') for line in open(src_file)]
    file_w = open(dst_file, 'w')
    dict_small_class = []
    dict_large_class = []
    for i in range(8000):
        dict_small_class.append([])
    for i in range(23):
        dict_large_class.append([])


    for i in xrange(len(lines)):
        cur_str = lines[i]
        split_str = cur_str.split('\t')
        if (int(split_str[1]) - 1) not in dict_large_class[int(split_str[2]) - 1]:
            dict_large_class[int(split_str[2]) - 1].append(int(split_str[1]) - 1)
        dict_small_class[int(split_str[1]) - 1].append(split_str[0])

    for i in range(batch_cnt):
        # step 1 : sample 2 big classes
        large_class_list = random.sample(range(23), 2)
        for j in large_class_list:
            # step 2 : sample 10 different small classes in each 2 big classes
            if (len(dict_large_class[j]) > 10):
                image_list = random.sample(dict_large_class[j], 10)
            else:
                image_list = dict_large_class[j]
            for k in image_list:
                # step 3 : sample all the images of each sampled sub-class
                for m in dict_small_class[k]:
                    #cur_line = "".join(m) + "\t" + str(k) + "\n"
                    cur_line = "".join(m) + " " + str(k) + "\n"
                    file_w.write(cur_line)
    file_w.close()

