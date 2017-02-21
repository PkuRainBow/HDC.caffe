cd ../script/CARS196/

"D:\users\v-yuhyua\cmake-caffe\caffe\build\tools\extract_features.exe" ^
		../../model/HDC/HDC_CARS196_iter_20000.caffemodel ^
		./HDC_deploy.prototxt loss1/classifier ^
		../../feature/HDC_CARS196_iter_20000_cls1 41 lmdb GPU 0

"D:\users\v-yuhyua\cmake-caffe\caffe\build\tools\extract_features.exe" ^
		../../model/HDC/HDC_CARS196_iter_20000.caffemodel ^
		./HDC_deploy.prototxt loss2/classifier ^
		../../feature/HDC_CARS196_iter_20000_cls2 41 lmdb GPU 0

"D:\users\v-yuhyua\cmake-caffe\caffe\build\tools\extract_features.exe" ^
		../../model/HDC/HDC_CARS196_iter_20000.caffemodel ^
		./HDC_deploy.prototxt loss3/classifier ^
		../../feature/HDC_CARS196_iter_20000_cls3 41 lmdb GPU 0