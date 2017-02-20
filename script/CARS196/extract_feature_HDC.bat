cd ../script/CARS196/

"../bin/extract_features.exe" ^
		../../model/HDC/HDC_CARS196_iter_20000.caffemodel ^
		./HDC_deploy.prototxt loss1/classifier ^
		./HDC_CARS196_iter_20000_cls1 41 lmdb GPU 0

"../bin/extract_features.exe" ^
		../../model/HDC/HDC_CARS196_iter_20000.caffemodel ^
		./HDC_deploy.prototxt loss2/classifier ^
		./HDC_CARS196_iter_20000_cls1 41 lmdb GPU 0

"../bin/extract_features.exe" ^
		../../model/HDC/HDC_CARS196_iter_20000.caffemodel ^
		./HDC_deploy.prototxt loss3/classifier ^d
		./HDC_CARS196_iter_20000_cls3 41 lmdb GPU 0