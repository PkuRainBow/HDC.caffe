import os, sys, getopt
from subprocess import Popen

def main(argv):
    dataset = ''
    choice = ''
    try:
        opts, args = getopt.getopt(argv, "hd:c:",["dataset=", "choice="])
    except getopt.GetoptError:
        print 'Error : hdc_train.py -d <dataset> -c <method>'
        print '    or: hdc_train.py --dataset=<dataset> --choice=<method>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print 'hdc_train.py -d <dataset> -c <method>'
            print 'or: hdc_train.py --dataset=<dataset> --choice=<method>'
            sys.exit()
        elif opt in ("-d", "--dataset"):
            dataset = arg
        elif opt in ("-c", "--choice"):
            choice = arg
    path_to_cur_python_file = os.path.dirname(__file__)
    print path_to_cur_python_file
    if choice == 'HDC':
        if dataset == 'stanford_products':
            path_to_batch_file = os.path.join(path_to_cur_python_file, '..','script','StanfordProducts',
                                              'run_HDC.bat')
            Popen(path_to_batch_file)
        elif dataset == 'cars196':
            path_to_batch_file = os.path.join(path_to_cur_python_file, '..', 'script', 'CARS196',
                                              'run_HDC.bat')
            Popen(path_to_batch_file)
        elif dataset == 'cub200':
            path_to_batch_file = os.path.join(path_to_cur_python_file, '..', 'script', 'CUB200',
                                              'run_HDC.bat')
            Popen(path_to_batch_file)
        elif dataset == 'deepfashion':
            Popen('../script/DeepFashion/run_HDC.bat')
        elif dataset == 'vehicleid':
            Popen('../script/VehicleID/run_HDC.bat')
        else:
            print 'Please choose the dataset from [stanford_products, cars196, cub200, deepfashion, vehicleid]'
    elif choice == 'Base':
        if dataset == 'cars196':
            Popen('../script/CARS196/run_Base.bat')
        elif dataset == 'cub200':
            Popen('../script/CUB200/run_Base.bat')
        else:
            print 'Please choose the dataset from [cars196, cub200]'
    elif choice == 'Hard':
        if dataset == 'cars196':
            Popen('../script/CARS196/run_Hard.bat')
        elif dataset == 'cub200':
            Popen('../script/CUB200/run_Hard.bat')
        else:
            print 'Please choose the dataset from [cars196, cub200]'
    else:
        print 'Please choose the method from [Base, Hard, HDC]'

if __name__ == "__main__":
    main(sys.argv[1:])

