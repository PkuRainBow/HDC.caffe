import sample_stanford_products
import sample_cars196
import sample_cub200
import sample_deepfashion
import sample_vehicleid
import util

import sys, getopt

def main(argv):
    dataset = ''
    try:
        opts, args = getopt.getopt(argv, "hd:",["dataset="])
    except getopt.GetoptError:
        print 'Error : hdc_process.py -d <dataset>'
        print '    or: hdc_process.py --dataset=<dataset>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print 'hdc_process.py -d <dataset>'
            print 'or: hdc_process.py --dataset=<dataset>'
            sys.exit()
        elif opt in ("-d", "--dataset"):
            dataset = arg

    if dataset == 'stanford_products':
        home_folder = '../data/stanford_products/'
        train_src_file = home_folder + 'Ebay_train.txt'
        test_src_file = home_folder + 'Ebay_test.txt'
        train_file = home_folder + 'train.txt'
        test_file = home_folder + 'test.txt'
        hdc_file = home_folder + 'hdc_train.txt'
        test_label_dat = home_folder + 'test_label.dat'
        class_cnt = 11318
        batch_cnt = 5000
        sample_stanford_products.preprocess_data(train_src_file, train_file)
        sample_stanford_products.preprocess_data(test_src_file, test_file)
        sample_stanford_products.create_training_data(train_src_file, hdc_file, class_cnt, batch_cnt) # sample the dataset for hdc training
        util.create_dat(test_file, test_label_dat)  # create the dat file for test
    elif dataset == 'cars196':
        home_folder = '../data/cars196/'
        src_file = home_folder + 'car_brand_all.txt'
        train_file = home_folder + 'train_8054.txt'
        test_file = home_folder + 'test_8131.txt'
        batch_cnt = 5000
        sample_cars196.preprocess_data(src_file, train_file, 'train')
        sample_cars196.preprocess_data(src_file, test_file, 'test')
        hdc_file = home_folder + 'hdc_train.txt'
        test_label_dat = home_folder + 'test_label.dat'
        sample_cars196.sample_data(train_file, hdc_file, 10, batch_cnt)
        util.create_dat(test_file, test_label_dat)  # create the dat file for test
    elif dataset == 'cub200':
        home_folder = '../data/cub200/'
        src_file = home_folder + 'birds_images.txt'
        train_file = home_folder + 'train_5864.txt'
        test_file = home_folder + 'test_5924.txt'
        batch_cnt = 5000
        sample_cub200.preprocess_data(src_file, train_file, 'train')
        sample_cub200.preprocess_data(src_file, test_file, 'test')
        hdc_file = home_folder + 'hdc_train.txt'
        test_label_dat = home_folder + 'test_label.dat'
        sample_cub200.sample_data(train_file, hdc_file, 10, batch_cnt)
        util.create_dat(test_file, test_label_dat)  # create the dat file for test
    elif dataset == 'deepfashion':
        home_folder = '../data/deepfashion/inshop/'
        src_file = home_folder + 'list_eval_partition.txt'
        train_file = home_folder + 'train.txt'
        query_file = home_folder + 'query.txt'
        gallery_file = home_folder + 'gallery.txt'
        hdc_file = home_folder + 'hdc_train.txt'
        query_label_dat = home_folder + 'query_label.dat'
        gallery_label_dat = home_folder + 'gallery_label.dat'
        crop_file = home_folder + 'list_bbox_inshop'
        batch_cnt = 5000
        input_dir = '../img/deepfashion/'
        output_dir = '../img/deepfashion_crop/'
        # sample_deepfashion.box_crop(input_dir, output_dir, crop_file)
        sample_deepfashion.partition_data(src_file, train_file, query_file, gallery_file)
        sample_deepfashion.sample_data(train_file, hdc_file, batch_cnt)
        util.create_dat(query_file, query_label_dat)
        util.create_dat(gallery_file, gallery_label_dat)
    elif dataset == 'vehicleid':
        home_folder = '../data/vehicleid/'
        src_train_file = home_folder + 'train_3cols.txt'
        src_val_file = home_folder + 'val.txt'
        hdc_file = home_folder + 'hdc_file.txt'
        test_file_1 = home_folder + 'test_800.txt'
        test_file_2 = home_folder + 'test_1600.txt'
        test_file_3 = home_folder + 'test_3200.txt'
        test_bat_1 = home_folder + 'test_800.bat'
        test_bat_2 = home_folder + 'test_1600.bat'
        test_bat_3 = home_folder + 'test_3200.bat'
        batch_cnt = 5000
        sample_vehicleid.sample(src_train_file, hdc_file, batch_cnt)
        util.create_dat(test_file_1, test_bat_1)  # create the dat file for test
        util.create_dat(test_file_2, test_bat_2)  # create the dat file for test
        util.create_dat(test_file_3, test_bat_3)  # create the dat file for test
    else:
        print 'Please choose the dataset from [stanford_products, cars196, cub200, deepfashion, vehicleid]'

if __name__ == "__main__":
    main(sys.argv[1:])
