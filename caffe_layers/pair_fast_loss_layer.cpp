/**
* Author : yhyuan@pku.edu.cn
* Date :   2016/06/09
* Update : 2016/11/05
*		   implement the adaptive margin mining method, we do not need to set the hard margin like 1 or others. we have no parameters.
*/

#include "caffe/layers/pair_fast_loss_layer.hpp"

namespace caffe {
	template <typename Dtype>
	void PairFastLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		vector<int> loss_shape_0(0);
		top[0]->Reshape(loss_shape_0);
		vector<int> loss_shape_1(0);
		top[1]->Reshape(loss_shape_1);
		vector<int> loss_shape_2(0);
		top[2]->Reshape(loss_shape_2);

		if (this->layer_param_.loss_weight_size() == 0) {
			this->layer_param_.add_loss_weight(Dtype(1));
			this->layer_param_.add_loss_weight(Dtype(0));
			this->layer_param_.add_loss_weight(Dtype(0));
		}

		CHECK_EQ(bottom[0]->width(), 1);
		CHECK_EQ(bottom[0]->height(), 1);
		CHECK_EQ(bottom[1]->channels(), 1);
		CHECK_EQ(bottom[1]->height(), 1);
		CHECK_EQ(bottom[1]->width(), 1);
		CHECK_EQ(bottom[1]->num(), bottom[0]->num());
		/** pre-compute the distance matrix for later computation
		pre-compute the diff matrix for convinience **/
		dist_matrix.Reshape(bottom[0]->num() * bottom[0]->num(), 1, 1, 1);
		pair_matrix.Reshape(bottom[0]->num() * bottom[0]->num(), 1, 1, 1);
		diff_matrix.Reshape(bottom[0]->num() * bottom[0]->num(), bottom[0]->channels(), 1, 1);
	}

	template <typename Dtype>
	void PairFastLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*> & bottom, const vector<Blob<Dtype>*> & top){
		const int channels = bottom[0]->channels();
		const int nums = bottom[0]->num();
		/** compute the diff_matrix and dist_matrix **/
		for (int i = 0; i < nums; ++i)
		{
			for (int j = i + 1; j < nums; ++j)
			{
				/** compute the diff matrix **/
				caffe_sub(channels, bottom[0]->cpu_data() + (i*channels), bottom[0]->cpu_data() + (j*channels),
					diff_matrix.mutable_cpu_data() + channels * (i * nums + j));
				caffe_sub(channels, bottom[0]->cpu_data() + (j*channels), bottom[0]->cpu_data() + (i*channels),
					diff_matrix.mutable_cpu_data() + channels * (j * nums + i));
				dist_matrix.mutable_cpu_data()[i * nums + j] = caffe_cpu_dot(channels,
					diff_matrix.cpu_data() + channels * (i * nums + j), diff_matrix.cpu_data() + channels * (i * nums + j));
				dist_matrix.mutable_cpu_data()[j * nums + i] = dist_matrix.mutable_cpu_data()[i * nums + j];
			}
		}

		/** compute the pair_matrix to record whether the loss is 0 at position (i,j), all init as 0 **/
		for (int i = 0; i < nums; i++)
		{
			for (int j = i + 1; j < nums; j++)
			{
				pair_matrix.mutable_cpu_data()[(i * nums + j)] = Dtype(0);
			}
		}

		/** the variable to record the loss and count information **/
		Dtype loss(0.0);
		Dtype hard_loss(0.0);
		Dtype neg_pair_count(0.0);
		Dtype pos_pair_count(0.0);
		Dtype triplet_count(0.0);

		Dtype correct_rank_count(0.0);
		Dtype triplet_rank_precision(0.0);
		Dtype margin = this->layer_param_.pair_fast_loss_param().margin();// alpha
		Dtype hard_ratio = this->layer_param_.pair_fast_loss_param().hard_ratio();// alpha
		Dtype factor = this->layer_param_.pair_fast_loss_param().factor();// alpha
		Dtype mode = this->layer_param_.pair_fast_loss_param().mode();

		const Dtype* bottom_label = bottom[1]->cpu_data();

		/** use the map to record accordding to the label info **/
		map<int, vector<int>> label_data_map;
		/** record the loss information into 2 vector, the pos pair loss and neg pair loss seperately **/
		vector<pair<float, pair<int, int>>> hard_loss_pos;
		vector<pair<float, pair<int, int>>> hard_loss_neg;

		int max_label = 0;
		for (int i = 0; i < bottom[0]->num(); i++) {
			const int label_value = static_cast<int>(bottom_label[i]);
			if (label_value > max_label)
				max_label = label_value;
			if (label_data_map.count(label_value) > 0) {
				label_data_map[label_value].push_back(i);
			}
			else{
				vector<int> tmp;
				tmp.push_back(i);
				label_data_map[label_value] = tmp;
			}
		}
		/** calculate the triplet precision **/

		for (auto const &ent1 : label_data_map)
		{
			int cur_size = label_data_map[ent1.first].size();
			if (cur_size < 2)
			{
				continue;
			}

			for (int j = 0; j < cur_size - 1; j++)
			{
				for (int k = j + 1; k < cur_size; k++)
				{
					int anc = label_data_map[ent1.first][j];
					int pos = label_data_map[ent1.first][k];
					for (auto const &ent2 : label_data_map)
					{
						if (ent1.first == ent2.first) continue;
						int m_size = label_data_map[ent2.first].size();
						for (int n = 0; n < m_size; n++)
						{
							triplet_count += Dtype(2);
							int neg = label_data_map[ent2.first][n];
							if (dist_matrix.cpu_data()[anc * nums + neg] > dist_matrix.cpu_data()[anc * nums + pos])   correct_rank_count += Dtype(1);
							if (dist_matrix.cpu_data()[pos * nums + neg] > dist_matrix.cpu_data()[pos * nums + anc])   correct_rank_count += Dtype(1);
						}
					}
				}
			}
		}

		/** loop all the possible pair-wise dataset and triplet dataset **/
		for (auto const &ent1 : label_data_map)
		{
			int cur_size = label_data_map[ent1.first].size();
			/** compute the same class pair-wise data loss **/
			for (int i = 0; i < cur_size - 1; i++)
			{
				//only consider neg pairs
				if (mode == 1) continue;
				for (int j = i + 1; j < cur_size; j++)
				{
					int pos_1 = label_data_map[ent1.first][i];
					int pos_2 = label_data_map[ent1.first][j];
					float loss_pos_pair = dist_matrix.cpu_data()[pos_1 * nums + pos_2];
					//float loss_pos_pair = max(dist_matrix.cpu_data()[pos_1 * nums + pos_2] - min_dist_class[ent1.first], Dtype(0.0));
					if (loss_pos_pair == 0) continue;
					pos_pair_count += Dtype(1);
					float tmp_pos_pair = loss_pos_pair;
					if (pos_1 < pos_2)	hard_loss_pos.push_back(std::make_pair(tmp_pos_pair, std::make_pair(pos_1, pos_2)));
					else hard_loss_pos.push_back(std::make_pair(tmp_pos_pair, std::make_pair(pos_2, pos_1)));
					loss += loss_pos_pair;
				}
			}
			/** compute the different class pair-wise data loss **/
			for (int i = 0; i < cur_size; i++)
			{
				//only consider pos pairs
				if (mode == 0) continue;
				int index_candidate = -1;
				for (auto const &ent2 : label_data_map)
				{
					//Dtype cur_margin = max(max_dist_class[ent1.first], max_dist_class[ent2.first]);
					index_candidate++;
					if (ent1.first == ent2.first) continue;
					int neg_size = label_data_map[ent2.first].size();
					for (int j = 0; j < neg_size; j++)
					{
						neg_pair_count += Dtype(1);
						int pos = label_data_map[ent1.first][i];
						int neg = label_data_map[ent2.first][j];
						Dtype loss_pos_neg = std::max(margin - dist_matrix.cpu_data()[pos * nums + neg], Dtype(0.0));
						float tmp_neg_pair = factor * loss_pos_neg;
						if (pos < neg)  hard_loss_neg.push_back(std::make_pair(tmp_neg_pair, std::make_pair(pos, neg)));
						else hard_loss_neg.push_back(std::make_pair(tmp_neg_pair, std::make_pair(neg, pos)));
						loss += tmp_neg_pair;
					}
				}
			}
		}
		/** sort all the hard sample set **/
		if (hard_ratio < 1) {
			sort(hard_loss_pos.begin(), hard_loss_pos.end(), [](const pair<float, pair<int, int>>& a, const pair<float, pair<int, int>>& b)
			{	return a.first > b.first; });
			sort(hard_loss_neg.begin(), hard_loss_neg.end(), [](const pair<float, pair<int, int>>& a, const pair<float, pair<int, int>>& b)
			{	return a.first > b.first; });
		}

		int pos_hard_cnt = pos_pair_count * hard_ratio;
		int neg_hard_cnt = neg_pair_count * hard_ratio;
		for (int i = 0; i < pos_hard_cnt; i++)  hard_loss += hard_loss_pos[i].first;
		for (int i = 0; i < neg_hard_cnt; i++)  hard_loss += hard_loss_neg[i].first;

		int all_hard_cnt = 0;
		for (int i = 0; i < pos_hard_cnt; i++) {
			if (hard_loss_pos[i].first > 0) {
				all_hard_cnt++;
				pair_matrix.mutable_cpu_data()[hard_loss_pos[i].second.first * nums + hard_loss_pos[i].second.second] = Dtype(1);
			}
		}
		for (int i = 0; i < neg_hard_cnt; i++) {
			if (hard_loss_neg[i].first > 0) {
				all_hard_cnt++;
				pair_matrix.mutable_cpu_data()[hard_loss_neg[i].second.first * nums + hard_loss_neg[i].second.second] = Dtype(1);
			}
		}
		hard_loss = hard_loss / all_hard_cnt;
		triplet_rank_precision = correct_rank_count / triplet_count;

		top[0]->mutable_cpu_data()[0] = hard_loss;
		top[1]->mutable_cpu_data()[0] = triplet_rank_precision;
		top[2]->mutable_cpu_data()[0] = all_hard_cnt;
	}

	template<typename Dtype>
	void PairFastLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		Dtype alpha = top[0]->cpu_diff()[0] / static_cast<Dtype>(top[2]->mutable_cpu_data()[0]);
		Dtype factor = this->layer_param_.pair_fast_loss_param().factor();
		const int channels = bottom[0]->channels();
		const int nums = bottom[0]->num();
		const int count = bottom[0]->count();

		if (propagate_down[1])
		{
			LOG(FATAL) << this->type()
				<< " Layer cannot backpropagate to label inputs.";
		}

		Dtype* bout = bottom[0]->mutable_cpu_diff();
		for (int i = 0; i < count; i++)
		{
			bout[i] = 0;
		}

		if (propagate_down[0])
		{
			for (int i = 0; i < nums; i++)
			{
				for (int j = i + 1; j < nums; j++)
				{
					if (pair_matrix.mutable_cpu_data()[i * nums + j] == Dtype(1))
					{
						// postive pair gradient update
						if ((bottom[1]->cpu_data()[i] == bottom[1]->cpu_data()[j]))
						{
							//Dtype loss_factor =  dist_matrix.cpu_data()[i * nums + j];
							caffe_cpu_axpby(
								channels,
								alpha,
								diff_matrix.cpu_data() + channels * (i * nums + j),
								Dtype(1),
								bout + (i*channels));
							caffe_cpu_axpby(
								channels,
								alpha,
								diff_matrix.cpu_data() + channels * (j * nums + i),
								Dtype(1),
								bout + (j*channels));
						}
						else
						{
							//Dtype loss_factor = abs(2 - dist_matrix.cpu_data()[i * nums + j]);
							caffe_cpu_axpby(
								channels,
								factor * alpha,
								diff_matrix.cpu_data() + channels * (j * nums + i),
								Dtype(1),
								bout + (i*channels));
							caffe_cpu_axpby(
								channels,
								factor * alpha,
								diff_matrix.cpu_data() + channels * (i * nums + j),
								Dtype(1),
								bout + (j*channels));
						}
					}
				}
			}
		}
	}
#ifdef CPU_ONLY
	STUB_GPU(PairFastLossLayer);
#endif

	INSTANTIATE_CLASS(PairFastLossLayer);
	REGISTER_LAYER_CLASS(PairFastLoss);

}//namespace caffe