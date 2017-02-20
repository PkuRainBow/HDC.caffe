cd ../script/CARS196/
SET init_weight=D:/users/v-yuhyua/fromGPU02/model/bvlc_googlenet.caffemodel
SET solver=./HDC_solver.prototxt

"D:\users\v-yuhyua\cmake-caffe\caffe\build\tools\caffe.exe" ^
train --solver=%solver% ^
--weights=%init_weight% --gpu=0