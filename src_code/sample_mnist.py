import caffe
import lmdb
import cv2
import numpy as np
from caffe.proto import caffe_pb2

lmdb_train = '../data/mnist/mnist_train_lmdb'
lmdb_test = '../data/mnist/mnist_test_lmdb'
lmdb_train_new = '../data/mnist/mnist_500'
# train_cnt_list = [50,50,50,2000,4000,10,100,1000,2000,4000]
train_cnt_list = [50,50,50,50,50,50,50,50,50,50]
lmdb_test_new = '../data/mnist/mnist_56789_unbalance'

# read from the training lmdb
lmdb_env_read = lmdb.open(lmdb_train)
lmdb_txn_read = lmdb_env_read.begin()
lmdb_cursor_read = lmdb_txn_read.cursor()
datum_read = caffe_pb2.Datum()
write_train_lmdb_env = lmdb.open(lmdb_train_new, map_size=int(1e8))
write_test_lmdb_env = lmdb.open(lmdb_test_new, map_size=int(1e8))
write_train_lmdb_txn = write_train_lmdb_env.begin(write=True)
write_test_lmdb_txn = write_test_lmdb_env.begin(write=True)
id_list = [-1] * 10
id_train = -1
id_test = -1
batch_size = 1000

for key, value in lmdb_cursor_read:
    datum_read.ParseFromString(value)
    label = datum_read.label
    data = caffe.io.datum_to_array(datum_read)
    if label < 5:
        id_list[label] += 1
        if id_list[label] < train_cnt_list[label]:
            id_train += 1
            keystr = '{:0>8d}'.format(id_train)
            datum = caffe.io.array_to_datum(data, label)
            datum_str = datum.SerializeToString()
            write_train_lmdb_txn.put(keystr, datum.SerializeToString())
            if (id_train + 1) % batch_size == 0:
                write_train_lmdb_txn.commit()
                write_train_lmdb_txn = write_train_lmdb_env.begin(write=True)
                print (id_train + 1)
    else:
        id_list[label] += 1
        if id_list[label] < train_cnt_list[label]:
            id_test += 1
            keystr = '{:0>8d}'.format(id_test)
            datum = caffe.io.array_to_datum(data, label) #ensure the label start from 0 to N
            write_test_lmdb_txn.put(keystr, datum.SerializeToString())
            if (id_test + 1) % batch_size == 0:
                write_test_lmdb_txn.commit()
                write_test_lmdb_txn = write_test_lmdb_env.begin(write=True)
                print (id_test + 1)

#read from the test lmdb
lmdb_env_read = lmdb.open(lmdb_test)
lmdb_txn_read = lmdb_env_read.begin()
lmdb_cursor_read = lmdb_txn_read.cursor()
datum_read = caffe_pb2.Datum()

for key, value in lmdb_cursor_read:
    datum_read.ParseFromString(value)
    label = datum_read.label
    data = caffe.io.datum_to_array(datum_read)
    if label < 5:
        id_list[label] += 1
        if id_list[label] < train_cnt_list[label]:
            id_train += 1
            keystr = '{:0>8d}'.format(id_train)
            datum = caffe.io.array_to_datum(data, label)
            write_train_lmdb_txn.put(keystr, datum.SerializeToString())
            if (id_train + 1) % batch_size == 0:
                write_train_lmdb_txn.commit()
                write_train_lmdb_txn = write_train_lmdb_env.begin(write=True)
                print (id_train + 1)
    else:
        id_list[label] += 1
        if id_list[label] < train_cnt_list[label]:
            id_test += 1
            keystr = '{:0>8d}'.format(id_test)
            datum = caffe.io.array_to_datum(data, label) #ensure the label start from 0 to N
            write_test_lmdb_txn.put(keystr, datum.SerializeToString())
            if (id_test + 1) % batch_size == 0:
                write_test_lmdb_txn.commit()
                write_test_lmdb_txn = write_test_lmdb_env.begin(write=True)
                print (id_test + 1)

if (id_train + 1) % batch_size != 0:
    write_train_lmdb_txn.commit()
    write_train_lmdb_txn = write_train_lmdb_env.begin(write=True)
    print 'last train batch'
    print (id_train + 1)

if (id_test + 1) % batch_size != 0:
    write_test_lmdb_txn.commit()
    write_test_lmdb_txn = write_test_lmdb_env.begin(write=True)
    print 'last test batch'
    print (id_test + 1)