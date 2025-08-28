
# Contrastive Learning for Semi-Supervised Deep Regression with Generalized Ordinal Rankings from Spectral Seriation



This is the official implementation of GCLSS (Generalized CLSS) and CLSS (NeurIPS 2023 ["Semi-Supervised Contrastive Learning for Deep Regression with Ordinal Rankings from Spectral Seriation"](https://openreview.net/forum?id=ij3svnPLzG)).

![GCLSS](GCLSS.png)

<br />
<br />

## Project Description
In this series of works, we have extended contrastive regression methods to allow both labeled and unlabeled data to be used in the semi-supervised setting, thereby reducing the dependence on costly annotations. 
* Particularly we construct the feature similarity matrix with both labeled and unlabeled samples in a mini-batch to reflect inter-sample relationships, and an accurate ordinal ranking of involved unlabeled samples can be recovered through spectral seriation algorithms.
* Labeled samples provides the regularization of the ordinal ranking with guidance from the ground-truth label information, making the ranking more reliable.
* We further utilize the dynamic programming algorithm to select robust features for the matrix construction.
* The recovered ordinal relationship is then used for contrastive learning on unlabeled samples, and we thus allow more data to be used for feature representation learning.
* We provide theoretical guarantees and empirical verification through experiments on various datasets, demonstrating that our method can surpass existing state-of-the-art semi-supervised deep regression methods.


## Implementation on operator learning, age estimation, and brain-age estimation

Implementations for the three tasks are provided in the separate folders. 


## Datasets and pre-trained weights

Required files are shared at:
https://hkustconnect-my.sharepoint.com/:f:/g/personal/wdaiaj_connect_ust_hk/Eu_ZWAv3ZCNHvNl4U24F-7sBnr9Ur57IWtbBHTnyIOGmdQ?e=VRNVGb 

Links are also available in the folders for the individual tasks.



## Notes
* Contact: WANG Ce (wangc79@mail.sysu.edu.cn) and DAI Weihang (wdai03@gmail.com)
<br />
<br />

## Citation
If this code is useful for your research, please consider citing:


```
@inproceedings{dai2023semi,
  title={Semi-Supervised Contrastive Learning for Deep Regression with Ordinal Rankings from Spectral Seriation},
  author={Dai, Weihang and Yao, DU and Bai, Hanru and Cheng, Kwang-Ting and Li, Xiaomeng},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}

```
