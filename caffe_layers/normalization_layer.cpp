/**
* normalization layer
*/

#include "caffe/layers/normalization_layer.hpp"


namespace caffe {
	template <typename Dtype>
	void NormalizationLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		//LOG(INFO) << "normalization set up ...";
		top[0]->ReshapeLike(*bottom[0]);
		squared_.ReshapeLike(*bottom[0]);
	}

	template <typename Dtype>
	void NormalizationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		//LOG(INFO) << "forward cpu mode  ...";
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		Dtype* squared_data = squared_.mutable_cpu_data();
		int n = bottom[0]->num();
		int d = bottom[0]->count() / n;
		caffe_sqr<Dtype>(n*d, bottom_data, squared_data);
		for (int i = 0; i<n; ++i) {
			Dtype normsqr = caffe_cpu_asum<Dtype>(d, squared_data + i*d);
			caffe_cpu_scale<Dtype>(d, pow(normsqr, -0.5), bottom_data + i*d, top_data + i*d);
		}
	}

	template <typename Dtype>
	void NormalizationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		//LOG(INFO) << "backward cpu mode  ...";
		const Dtype* top_diff = top[0]->cpu_diff();
		const Dtype* top_data = top[0]->cpu_data();
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		int n = top[0]->num();
		int d = top[0]->count() / n;

		for (int i = 0; i < n; ++i) {
			Dtype a = caffe_cpu_dot(d, top_data + i*d, top_diff + i*d);
			caffe_cpu_scale(d, a, top_data + i*d, bottom_diff + i*d);
			caffe_sub(d, top_diff + i*d, bottom_diff + i*d, bottom_diff + i*d);
			a = caffe_cpu_dot(d, bottom_data + i*d, bottom_data + i*d);
			caffe_cpu_scale(d, Dtype(pow(a, -0.5)), bottom_diff + i*d, bottom_diff + i*d);
		}
	}
#ifdef CPU_ONLY
	STUB_GPU(NormalizationLayer);
#endif

	INSTANTIATE_CLASS(NormalizationLayer);
	REGISTER_LAYER_CLASS(Normalization);
}//namespace caffe
