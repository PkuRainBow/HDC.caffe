/**
* Author : yhyuan@pku.edu.cn
* Date :   2016/06/09
*/

#ifndef CAFFE_PAIR_FAST_LOSS_LAYER_HPP_
#define CAFFE_PAIR_FAST_LOSS_LAYER_HPP_

#include <vector>
#include <algorithm>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
	/**
	* @brief computes the triplet fast loss
	*/
	template <typename Dtype>
	class PairFastLossLayer : public LossLayer<Dtype> {
	public:
		explicit PairFastLossLayer(const LayerParameter& param) :LossLayer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual inline int ExactNumBottomBlobs() const{ return 2; }//#num of bottom blobs.
		virtual inline int ExactNumTopBlobs() const { return 3; } //one record the loss , the other record the pair-wise neg precision
		virtual inline const char* type() const { return "PairFastLoss"; }
		/**
		*Unlike most loss layers, in the TripletFastLossLayer we can backpropagate to the first three inputs.
		*/
		virtual inline bool AllowForceBackward(const int bottom_index) const {
			return bottom_index != 1;
		}
	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom);

		Blob<Dtype> dist_matrix;
		Blob<Dtype> diff_matrix;
		Blob<Dtype> pair_matrix;
		PairFastLossParameter param_;

		//map<int, Dtype> max_dist_class;
		//map<int, Dtype> min_dist_class;
	};
}  // namespace caffe
#endif  // CAFFE_PAIR_FAST_LOSS_LAYER_HPP_
