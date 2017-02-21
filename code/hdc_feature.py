import os, sys, getopt
from subprocess import Popen

def main(argv):
    dataset = ''
    choice = ''
    try:
        opts, args = getopt.getopt(argv, "hd:c:",["dataset=", "choice="])
    except getopt.GetoptError:
        print 'Error : hdc_feature.py -d <dataset> -c <method>'
        print '    or: hdc_feature.py --dataset=<dataset> --choice=<method>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print 'hdc_feature.py -d <dataset> -c <method>'
            print 'or: hdc_feature.py --dataset=<dataset> --choice=<method>'
            sys.exit()
        elif opt in ("-d", "--dataset"):
            dataset = arg
        elif opt in ("-c", "--choice"):
            choice = arg

    path_to_cur_python_file = os.path.dirname(__file__)
    print path_to_cur_python_file
    if choice == 'HDC':
        if dataset == 'stanford_products':
            path_to_batch_file = os.path.join(path_to_cur_python_file, '..', 'script', 'StanfordProducts',
                                              'extract_feature_HDC.bat')
            Popen(path_to_batch_file)
        elif dataset == 'cars196':
            path_to_batch_file = os.path.join(path_to_cur_python_file, '..', 'script', 'CARS196',
                                              'extract_feature_HDC.bat')
            Popen(path_to_batch_file)
        elif dataset == 'cub200':
            path_to_batch_file = os.path.join(path_to_cur_python_file, '..', 'script', 'CUB200',
                                              'extract_feature_HDC.bat')
            Popen(path_to_batch_file)
        elif dataset == 'deepfashion':
            path_to_batch_file = os.path.join(path_to_cur_python_file, '..', 'script', 'DeepFashion',
                                              'extract_feature_HDC.bat')
            Popen(path_to_batch_file)
        else:
            print 'Please choose the dataset from [stanford_products, cars196, cub200, deepfashion]'
    elif choice == 'Base':
        if dataset == 'cars196':
            path_to_batch_file = os.path.join(path_to_cur_python_file, '..', 'script', 'CARS196',
                                              'extract_feature_Base.bat')
            Popen(path_to_batch_file)
        elif dataset == 'cub200':
            path_to_batch_file = os.path.join(path_to_cur_python_file, '..', 'script', 'CUB200',
                                              'extract_feature_Base.bat')
            Popen(path_to_batch_file)
        else:
            print 'Please choose the dataset from [cars196, cub200]'
    elif choice == 'Hard':
        if dataset == 'cars196':
            path_to_batch_file = os.path.join(path_to_cur_python_file, '..', 'script', 'CARS196',
                                              'extract_feature_Hard.bat')
            Popen(path_to_batch_file)
        elif dataset == 'cub200':
            path_to_batch_file = os.path.join(path_to_cur_python_file, '..', 'script', 'CUB200',
                                              'extract_feature_Hard.bat')
            Popen(path_to_batch_file)
        else:
            print 'Please choose the dataset from [cars196, cub200]'
    else:
        print 'Please choose the method from [Base, Hard, HDC]'

if __name__ == "__main__":
    main(sys.argv[1:])

