## Derivative Manipulation: Example Weighting via Emphasis Density Funtion in the context of DL

* [Paper](https://arxiv.org/pdf/1905.11233.pdf)
* [Slide: Example Weighting for Deep Representation Learning using IMAE and DM](./IMAE_DM_V03_17052020_at_KAUST_VISION-CAIR_GROUP.pdf)


#### :+1: Glad to know that our recent papers have inspired an ICML 2020 paper: [Normalized Loss Functions for Deep Learning with Noisy Labels](https://arxiv.org/pdf/2006.13554.pdf)
* [https://www.reddit.com/r/MachineLearning/comments/hjlayq/r_open_discussion_on_deep_robustness_please/](https://www.reddit.com/r/MachineLearning/comments/hjlayq/r_open_discussion_on_deep_robustness_please/)
* This paper is still working on improving loss functions.

* Instead in our work, we go deeper and propose a much more flexible framework **to design the derivative straightforwardly without deriving it from a loss function, [is not it very cool?](https://arxiv.org/pdf/1905.11233.pdf)**


#### :+1: [Code is available now!](https://xinshaoamoswang.github.io/blogs/2020-06-14-code-releasing/)


### Citation
Please kindly cite us if you find our work useful and inspiring.

```bash
@article{wang2019derivative,
  title={Derivative Manipulation for General Example Weighting},
  author={Wang, Xinshao and Kodirov, Elyor and Hua, Yang and Robertson, Neil},
  journal={arXiv preprint arXiv:1905.11233},
  year={2019}
}
```



### Questions
* What is the right way to conduct example weighting?  
* How do you understand loss functions and so-called theorems on them? 

### The importance of example weighting
* Focusing on informative examples;
* Addressing Label Noise
* Addressing sample imbalance (long-tailed data distribution)
* Faster convergence
* ...

### Rethinking 
* Rethinking Proposed Example Weighting Schemes;
* Rethinking Existing Robustness Theorems on Loss
Functions;
* ...


### Representative Questions on Why does it work? (#Tag: From ICML 2020, thanks to the reviewer)

0.**Lack of theoretical backing**:
The idea of using example weighting to train robust models has been gaining a lot of traction, including prior approaches that automatically learn the re-weighting function to enable robustness to noise.

I find the high-level idea of interpreting example re-weighting as manipulation of the loss gradients to be interesting, and that this then allows us to alter the magnitude of the gradients at the example level. Moreover, the paper also does a nice job of designing the gradient manipulation scheme based on the "probability p_i assigned by a model to the true label", and arguing that points that end up with a lower p_i represent noisy/outlier examples and need to assigned lower a weight.

My main concern is that this scheme is one among several heuristics one could use and does not have a strong theoretical backing. For example, what is the effective optimization problem being solved with the manipulated gradients? Can you say something about the robustness of the resulting trained model under specific assumptions on the noise in the data?

1.**Concerns about weighting scheme (local changes vs global effect)**:
I have two main comments/concern about the re-weighting scheme:

Firstly, changes to the local gradients changes the landscape of the objective being optimized. I'm not sure if it's obvious that the effective optimization being solved does indeed train the model more robustly.

The authors are correct that they only change the magnitude of the gradient per-example and not its direction. However, when averaged over multiple examples, doesn't this effectively alter the direction of the average gradient, and thus alter the landscape of the objective being optimized?

Secondly, the intuition that examples with a low "probability p_i" might be outliers and should be assigned lower weights holds true for a fully trained model. However, early on in the training process, it's quite possible that even easy examples receive a lower p_i. Wouldn't your example re-weighting scheme result in such examples being ignored as a result of the gradient magnitudes being decreased by your scheme?

2.**In other words, it's not entirely clear how local manipulations to gradients effect the overall objective being optimized, and this is where some theoretical results showcasing the global effects of the proposed local manipulations would provide greater credibility to the method. Feel free to correct me if I have misunderstood your scheme, or if this is concern of mine is already addressed in your paper.**


#### **Personal Answer (I will provide more detailed response later)**: 
* In this paper, we do not provide new theorems. Instead, our framework challenges existing theorems on loss functions, e.g., a loss function is okay to be non-symmetric, unbounded, or even non-differentiable. 
 
* Our DM is heuristic, simple and useful in practice, without taking into consideration any assumptions on the training data. 

* **How local manipulations to gradients effect the overall objective being optimized?**: 
Local manipulation is the right thing we can do in deep learning optimsiation after we have chosen the network architecture. Is not it?
Commom practices are: (1) Changing loss functions; (2) Output regularisation (softer targets, confidence penalty, etc)
 


## Introduction

* **Why Derivative Manipulation:** We propose derivative manipulation (DM) for
training accurate and robust softmax-based deep
neural networks. Robust loss functions and example weighting are two popular solutions. Why
derivative manipulation? (1) In gradient-based
optimisation, derivative other than loss value has
direct impact on the update of a model. Therefore, we manipulate derivative directly, which is
more straightforward than designing loss functions. (2) The loss’s derivative of an example
defines how much impact its has on the update of
a model, which can be interpreted as its ‘weight’.
Therefore, a loss function’s derivative magnitude
function can be understood as a weighting scheme.
Manipulating derivative is to adjust the weighting
scheme.

* **How to Manipulate:**  DM simply modifies derivative’s magnitude, including transformation and normalisation, after
which derivative’s magnitude function is termed
emphasis density function (EDF). An EDF is a formula expression of an example weighting scheme.
We educe many options for EDFs from probability density functions (PDFs). We demonstrate
the effectiveness of DM empirically by extensive
experiments.

* **Rethinking Existing Robustness Theorems on Loss
Functions**: Given a loss function, when an extremely
large loss value occurs, its corresponding derivative’s magnitude may be small and ignorable. In this case, from the
loss value’s perspective, this loss function is neither symmetric or bounded, thus being non-robust. However, from the
derivative’s viewpoint, its gradient is so small that it almost
has no effect on the update of a model. Therefore, the loss
function is robust. There are many empirical evidences for
justification: **(1) (Rolnick et al., 2017) demonstrated that a
deep model trained by CCE is actually robust to massive
label noise; (2) We find that CCE is very competitive versus
MAE, MSE and GCE in our experiments where loss is the
only variable. However, CCE is neither bounded nor symmetric. But its derivative function is bounded as shown in
Figure 1a.**

* **Rethinking Proposed Example Weighting Schemes**: In prior work where a new example weighting is proposed,
**there is no analysis on the interaction between it and example weighting coming from a loss function**. However, the example weighting defined by a loss’s derivative also varies in different loss functions, as shown in Figure 1a. Therefore, the interaction between a proposed example weighting scheme and the one from a loss function
may be either positive or negative. We compare with some
recently proposed example weighting algorithms in Table 3.




## Additional Information

[More comments and comparison with related work](https://www.researchgate.net/publication/333418661_Emphasis_Regularisation_by_Gradient_Rescaling_for_Training_Deep_Neural_Networks_with_Noisy_Labels/comments)




## Extremely Simple and Effective

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

## Tables and Figures

Please see [our paper](https://arxiv.org/pdf/1905.11233.pdf): 

<p float="left">
  <img src="./figs/1.png" width="400">
  <img src="./figs/2.png" width="400">
  <img src="./figs/3.png" width="400">
  <img src="./figs/4.png" width="400">
  <img src="./figs/6.png" width="400">
  <img src="./figs/5.png" width="800">
</p>


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
