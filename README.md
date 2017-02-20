# Hard-Aware-Deeply-Cascaded-Embedding_release

This repository has the source code for the paper "Hard-Aware-Deeply-Cascaded-Embedding_release"(submit to CVPR-2017). This paper is available
 on [arXiv](https://arxiv.org/abs/1611.05720). For the loss layer implementation, look at the folder caffe_layers.
 

## Citing this work
If you find this work useful in your research, please consider citing :

    @article{yuan2016HDC,
      title={Hard-Aware Deeply Cascaded Embedding},
      author={Yuan, Yuhui and Yang, Kuiyuan and Zhang, Chao},
      journal={arXiv preprint arXiv:1611.05720},
      year={2016}
    }
    
## Installation
1. Install [Caffe](https://github.com/BVLC/caffe) (including the python interface if you want to use the test code) 
2. Add the "NormalizationLayer" and "PairFastLossLayer" to the caffe.
please add the following lines the **caffe.proto** :
      optional PairFastLossParameter pair_fast_loss_param = 154; //change it to a number according to your version
      message PairFastLossParameter {
        //margin for dissimilar pair
        optional float margin = 1 [default = 1.0];
        optional float hard_ratio = 2 [default = 1.0];
        optional float only_pos = 3 [default = 0];
      }
      
## Prerequisites
1. caffe (python interface)
2. matplotlib,cPickle,numpy,lmdb
3. We assumed that you are using Windows (you can rewrite the *.bat to *.sh if you choose Linux or MacOs)

## Usage 
1. Download the datasets : 
