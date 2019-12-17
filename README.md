# Reweight training examples to reduce overfitting and improve generalisation
Project page for [Emphasis Regularisation by Gradient Rescaling ](https://arxiv.org/pdf/1905.11233.pdf). 

General applicability: Label noise (semantic noise),  outliers,  heavy perceptual data noise, etc. 

**It is reasonable to assume that there is semantic noise in large-scale training datasets**: 
* Class labels may be missing. 
* The labelling process may be subjective. 



Label noise is one of the most explicit cases where some observations and their labels are not matched in the training data. In this case, it is quite crucial to make your models learn meaningful patterns instead of errors.

## Code is available now

The code is simple in several lines. Please check CCE layer and GR layer for the exact differences. 

The key codes are presented as follows:
```
  const Dtype lambda_p = this->layer_param_.loss_param().lambda_p();
  const Dtype scale = this->layer_param_.loss_param().scale();

  //non-linear transformation - exp
  inline Dtype softmaxT(const Dtype x, const Dtype T, const Dtype base) 
  {
    return pow( base, T * x );
  }
```

```
//Forward computation p_i = prob_data[i * dim + label_value * inner_num_ + j]
  //1. compute the weight value of one example
  Dtype temp = softmaxT(lambda_p, Dtype(1), 
          prob_data[i * dim + label_value * inner_num_ + j]);
  const Dtype weight_value = softmaxT(scale * temp, 
      		(1 - prob_data[i * dim + label_value * inner_num_ + j] ), 
          Dtype(2.718281828));
  
  // 2. The loss is scaled for output. 
  // This scaling is not important and only for output reference. It has no impact on gradient computation and back-propagation. 
  loss -= weight_value * log(std::max(prob_data[i * dim + label_value * inner_num_ + j], Dtype(FLT_MIN)));

  // 3. For normalisation purpose
  sum_weight += weight_value; 
  top[0]->mutable_cpu_data()[0] = loss / sum_weight;
```

```
//Backward computation: 
//Eq. (6) and (7) in https://arxiv.org/pdf/1905.11233.pdf
  
  // 1.  Gradient rescaling 
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
              // Remove built-in / intrinsic weighting of CCE
              // CCE has built-in weighting for training examples, which is well studied in the literature. 
              bottom_diff[i * dim + c * inner_num_ + j] /= ( 2 - 2 * prob_data[i * dim + label_value * inner_num_ + j] + Dtype(1e-6) );

              // New weight is imposed.
              bottom_diff[i * dim + c * inner_num_ + j] *= weight_value;
          }

  // 2. For normalisation purpose
  Dtype loss_weight = top[0]->cpu_diff()[0] / sum_weight;
```


## Introduction

* **Main contribution:** Intuitively and principally, we claim that two basic factors, what examples
get higher weights (emphasis focus) and how large variance over examples’ weights (emphasis
spread), should be babysit simultaneously when it comes to sample differentiation and reweighting.
Unfortunately, these two intuitive and indispensable factors are not studied together in the literature.

* What training examples should be focused and how much more should they be emphasised when training DNNs under label noise? 

  **When noise rate is higher, we can improve a model's robustness by focusing on relatively less difficult examples.** 

[More comments and comparison with related work](https://www.researchgate.net/publication/333418661_Emphasis_Regularisation_by_Gradient_Rescaling_for_Training_Deep_Neural_Networks_with_Noisy_Labels/comments)

[Paper reading about outlier detection and robust inference](https://drive.google.com/file/d/1fU3N_u-_puOwEbupK6aOENerP2S45tZX/view?usp=sharing)


## Effective (Qualitative and Quantitative Results)

Please see [our paper](https://arxiv.org/pdf/1905.11233.pdf): 

* Outperform existing work on synthetic label noise;
* Outperform existing work on unknown real-world noise. 

<p float="left">
  <img src="./figs/Figure1.png" width="800">
  <img src="./figs/Table1.png" width="800">
  <img src="./figs/Figure2.png" width="800">
  <img src="./figs/Table4.png" width="800">
  <img src="./figs/Table5.png" width="800">
  <img src="./figs/Table6.png" width="800">
  <img src="./figs/Table7.png" width="800">
  <img src="./figs/Table9.png" width="800">
</p>


## Extremely Simple

**Without advanced training strategies**: e.g., 

  a. Iterative retraining on gradual data correction

  b. Training based on carefully-designed curriculums

  ...

**Without using extra networks**: e.g., 

  a. Decoupling" when to update" from" how to update"  
  
  b. Co-teaching: Robust training of deep neural networks with extremely noisy labels
  
  c. Mentornet: Learning datadriven curriculum for very deep neural networks on corrupted labels

  ...

**Without using extra validation sets for model optimisation**: e.g., 

  a.  Learning to reweight examples for
robust deep learning

  b. Mentornet: Learning datadriven curriculum for very deep neural networks on corrupted labels

  c. Toward robustness against label noise in training deep discriminative neural networks

  d. Learning
from noisy large-scale datasets with minimal supervision.

  e. Learning from
noisy labels with distillation. 

  f. Cleannet: Transfer learning for
scalable image classifier training with label noise

  ...

**Without data pruning**: e.g., 

  a. Generalized cross entropy loss for training deep neural networks
with noisy labels.   
  ...

**Without relabelling**: e.g.,

  a. A semi-supervised two-stage approach
to learning from noisy labels

  b. Joint optimization framework for learning with noisy labels

  ...





## Citation
Please kindly cite us if you find our work useful and inspiring.

```bash
@article{wang2019emphasis,
  title={Emphasis Regularisation by Gradient Rescaling for Training Deep Neural Networks Robustly},
  author={Wang, Xinshao and Hua, Yang and Kodirov, Elyor and Robertson, Neil},
  journal={arXiv preprint arXiv:1905.11233},
  year={2019}
}
```



## References

* Eran Malach and Shai Shalev-Shwartz. Decoupling" when to update" from" how to update". In
NIPS, 2017.

* Bo Han, Quanming Yao, Xingrui Yu, Gang Niu, Miao Xu, Weihua Hu, Ivor Tsang, and Masashi
Sugiyama. Co-teaching: Robust training of deep neural networks with extremely noisy labels. In
NIPS, 2018

* Lu Jiang, Zhengyuan Zhou, Thomas Leung, Li-Jia Li, and Li Fei-Fei. Mentornet: Learning datadriven curriculum for very deep neural networks on corrupted labels. In ICML, 2018.

* Mengye Ren, Wenyuan Zeng, Bin Yang, and Raquel Urtasun. Learning to reweight examples for
robust deep learning. In ICML, 2018.


* Arash Vahdat. Toward robustness against label noise in training deep discriminative neural networks.
In NIPS, 2017.


* Andreas Veit, Neil Alldrin, Gal Chechik, Ivan Krasin, Abhinav Gupta, and Serge Belongie. Learning
from noisy large-scale datasets with minimal supervision. In CVPR, 2017.

* Yuncheng Li, Jianchao Yang, Yale Song, Liangliang Cao, Jiebo Luo, and Li-Jia Li. Learning from
noisy labels with distillation. In ICCV, 2017.

* Kuang-Huei Lee, Xiaodong He, Lei Zhang, and Linjun Yang. Cleannet: Transfer learning for
scalable image classifier training with label noise. In CVPR, 2018.

* Zhilu Zhang and Mert R Sabuncu. Generalized cross entropy loss for training deep neural networks
with noisy labels. In NIPS, 2018.

* Yifan Ding, Liqiang Wang, Deliang Fan, and Boqing Gong. A semi-supervised two-stage approach
to learning from noisy labels. In WACV, 2018.


* Hwanjun Song, Minseok Kim, and Jae-Gil Lee. Selfie: Refurbishing unclean samples for robust
deep learning. In ICML, 2019.


* Daiki Tanaka, Daiki Ikami, Toshihiko Yamasaki, and Kiyoharu Aizawa. Joint optimization framework for learning with noisy labels. In CVPR, 2018.
