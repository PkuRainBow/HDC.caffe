import os, sys, getopt
import test_stanford_products
import test_cars196
import test_cub200
import test_deep_fashion

def main(argv):
    dataset = ''
    choice = ''
    try:
        opts, args = getopt.getopt(argv, "hd:c:",["dataset=", "choice="])
    except getopt.GetoptError:
        print 'Error : hdc_test.py -d <dataset> -c <method>'
        print '    or: hdc_test.py --dataset=<dataset> --choice=<method>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print 'hdc_test.py -d <dataset> -c <method>'
            print 'or: hdc_test.py --dataset=<dataset> --choice=<method>'
            sys.exit()
        elif opt in ("-d", "--dataset"):
            dataset = arg
        elif opt in ("-c", "--choice"):
            choice = arg
    if choice == 'HDC':
        feature_dim = 384
        if dataset == 'stanford_products':
            query_lmdb_path_1 = '../feature/HDC_StanfordProducts_iter_60000_cls1'
            query_lmdb_path_2 = '../feature/HDC_StanfordProducts_iter_60000_cls2'
            query_lmdb_path_3 = '../feature/HDC_StanfordProducts_iter_60000_cls3'
            query_label_file = '../data/stanford_products/test_label.dat'
            data_lmdb_path_1 = query_lmdb_path_1
            data_lmdb_path_2= query_lmdb_path_2
            data_lmdb_path_3 = query_lmdb_path_3
            data_label_file = query_label_file
            data_cnt = 60499
            top_cnt = 1001
            test_stanford_products.eval(query_lmdb_path_1, query_lmdb_path_2, query_lmdb_path_3, query_label_file,
                                        data_lmdb_path_1, data_lmdb_path_2, data_lmdb_path_3, data_label_file ,
                                        data_cnt, feature_dim, top_cnt)
        elif dataset == 'cars196':
            query_lmdb_path_1 = '../feature/HDC_CARS196_iter_20000_cls1'
            query_lmdb_path_2 = '../feature/HDC_CARS196_iter_20000_cls2'
            query_lmdb_path_3 = '../feature/HDC_CARS196_iter_20000_cls3'
            query_label_file = '../data/stanford_products/test_label.dat'
            data_lmdb_path_1 = query_lmdb_path_1
            data_lmdb_path_2= query_lmdb_path_2
            data_lmdb_path_3 = query_lmdb_path_3
            data_label_file = query_label_file
            data_cnt = 8131
            top_cnt = 32
            test_cars196.eval(query_lmdb_path_1, query_lmdb_path_2, query_lmdb_path_3, query_label_file,
                                        data_lmdb_path_1, data_lmdb_path_2, data_lmdb_path_3, data_label_file ,
                                        data_cnt, feature_dim, top_cnt)
        elif dataset == 'cub200':
            query_lmdb_path_1 = '../feature/HDC_CUB200_iter_20000_cls1'
            query_lmdb_path_2 = '../feature/HDC_CUB200_iter_20000_cls2'
            query_lmdb_path_3 = '../feature/HDC_CUB200_iter_20000_cls3'
            query_label_file = '../data/stanford_products/test_label.dat'
            data_lmdb_path_1 = query_lmdb_path_1
            data_lmdb_path_2= query_lmdb_path_2
            data_lmdb_path_3 = query_lmdb_path_3
            data_label_file = query_label_file
            data_cnt = 5924
            top_cnt = 32
            test_cub200.eval(query_lmdb_path_1, query_lmdb_path_2, query_lmdb_path_3, query_label_file,
                              data_lmdb_path_1, data_lmdb_path_2, data_lmdb_path_3, data_label_file,
                              data_cnt, feature_dim, top_cnt)

        elif dataset == 'deepfashion':
            query_lmdb_path = '../feature/HDC_deepfashion_query'
            query_label_file = '../data/deepfashion/inshop/query_label.dat'
            data_lmdb_path = '../feature/HDC_deepfashion_gallery'
            data_label_file = '../data/deepfashion/inshop/gallery_label.dat'
            gallery_cnt = 12612
            query_cnt = 14218
            top_cnt = 50
            test_deep_fashion.eval(query_lmdb_path, query_label_file, data_lmdb_path, data_label_file, query_cnt, gallery_cnt, feature_dim,top_cnt)
        else:
            print 'Please choose the dataset from [stanford_products, cars196, cub200, deepfashion]'
    elif choice == 'Base':
        feature_dim = 128
        if dataset == 'cars196':
            query_lmdb_path = '../feature/Base_CARS196_iter_20000'
            query_label_file = '../data/cars196/test_label.dat'
            data_lmdb_path = query_lmdb_path
            data_label_file = query_label_file
            data_cnt = 8131
            top_cnt = 32
            test_cars196.eval(query_lmdb_path, query_label_file, data_lmdb_path, data_label_file, data_cnt, feature_dim,top_cnt)
        elif dataset == 'cub200':
            query_lmdb_path = '../feature/Base_cub200'
            query_label_file = '../data/cub200/test_label.dat'
            data_lmdb_path = query_lmdb_path
            data_label_file = query_label_file
            data_cnt = 5924
            top_cnt = 32
            test_cub200.eval(query_lmdb_path, query_label_file, data_lmdb_path, data_label_file, data_cnt, feature_dim, top_cnt)
        else:
            print 'Please choose the dataset from [cars196, cub200]'
    elif choice == 'Hard':
        feature_dim = 128
        if dataset == 'cars196':
            query_lmdb_path = '../feature/Hard_cars196'
            query_label_file = '../data/cars196/test_label.dat'
            data_lmdb_path = query_lmdb_path
            data_label_file = query_label_file
            data_cnt = 8131
            top_cnt = 32
            test_cars196.eval(query_lmdb_path, query_label_file, data_lmdb_path, data_label_file, data_cnt, feature_dim,top_cnt)
        elif dataset == 'cub200':
            query_lmdb_path = '../feature/Hard_cub200'
            query_label_file = '../data/cub200/test_label.dat'
            data_lmdb_path = query_lmdb_path
            data_label_file = query_label_file
            data_cnt = 5924
            top_cnt = 32
            test_cub200.eval(query_lmdb_path, query_label_file, data_lmdb_path, data_label_file, data_cnt, feature_dim, top_cnt)
        else:
            print 'Please choose the dataset from [cars196, cub200]'
    else:
        print 'Please choose the method from [Base, Hard, HDC]'

if __name__ == "__main__":
    main(sys.argv[1:])