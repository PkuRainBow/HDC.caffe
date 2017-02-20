cd ../script/StanfordProducts/

"../bin/extract_features.exe" ^
		../../model/HDC/HDC_StanfordProducts_iter_60000.caffemodel ^
		./HDC_deploy.prototxt loss1/classifier ^
		./HDC_StanfordProducts_iter_60000_cls1 303 lmdb GPU 0

"../bin/extract_features.exe" ^
		../../model/HDC/HDC_StanfordProducts_iter_60000.caffemodel ^
		./HDC_deploy.prototxt loss2/classifier ^
		./HDC_StanfordProducts_iter_60000_cls2 303 lmdb GPU 0

"../bin/extract_features.exe" ^
		../../model/HDC/HDC_StanfordProducts_iter_60000.caffemodel ^
		./HDC_deploy.prototxt loss3/classifier ^
		./HDC_StanfordProducts_iter_60000_cls3 303 lmdb GPU 0