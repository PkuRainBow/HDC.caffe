cd ../script/CUB200/

"../bin/extract_features.exe" ^
		../../model/HDC/HDC_CUB200_iter_60000.caffemodel ^
		./HDC_deploy.prototxt loss1/classifier ^
		./HDC_CUB200_iter_20000_cls1 30 lmdb GPU 0

"../bin/extract_features.exe" ^
		../../model/HDC/HDC_CUB200_iter_60000.caffemodel ^
		./HDC_deploy.prototxt loss2/classifier ^
		./HDC_CUB200_iter_20000_cls2 30 lmdb GPU 0

"../bin/extract_features.exe" ^
		../../model/HDC/HDC_CUB200_iter_60000.caffemodel ^
		./HDC_deploy.prototxt loss3/classifier ^
		./HDC_CUB200_iter_20000_cls3 30 lmdb GPU 0
		