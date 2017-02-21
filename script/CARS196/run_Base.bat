cd ../script/CARS196/
SET init_weight=D:/users/v-yuhyua/fromGPU02/model/bvlc_googlenet.caffemodel
SET solver=./Base_solver.prototxt

"D:\users\t-yuhyua\tools\caffe.exe" ^
train --solver=%solver% ^
--weights=%init_weight% --gpu=0,1,2,3