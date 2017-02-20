/**
* Author : yhyuan@pku.edu.cn
* Date :   2016/06/09
*/

#include "caffe/layers/pair_fast_loss_layer.hpp"


namespace caffe {

	template <typename Dtype>
	void PairFastLossLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*> & bottom, const vector<Blob<Dtype>*> & top){
		Forward_cpu(bottom, top);
	}

	template<typename Dtype>
	void PairFastLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		Backward_cpu(top, propagate_down, bottom);
	}
	INSTANTIATE_LAYER_GPU_FUNCS(PairFastLossLayer);
}//namespace caffe