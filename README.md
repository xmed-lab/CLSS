
# Contrastive Learning for Semi-Supervised Deep Regression with Generalized Ordinal Rankings from Spectral Seriation



This is the official implementation of GCLSS (Generalized CLSS) and CLSS (NeurIPS 2023 ["Semi-Supervised Contrastive Learning for Deep Regression with Ordinal Rankings from Spectral Seriation"](https://openreview.net/forum?id=ij3svnPLzG)).

![GCLSS](/docs/GCLSS.png)

<br />
<br />

## Project Description
In this series of works, we have extended contrastive regression methods to allow both labeled and unlabeled data to be used in the semi-supervised setting, thereby reducing the dependence on costly annotations:
- We construct the feature similarity matrix with both labeled and unlabeled samples to reflect inter-sample relationships, and an accurate ordinal ranking can be recovered through spectral seriation algorithms;
- Labeled samples provides the regularization with guidance from the ground-truth label information, making the ranking more reliable;
- We utilize the dynamic programming algorithm to select robust features;
- The recovered ordinal relationship is used for contrastive learning on unlabeled samples;
- We provide theoretical guarantees and empirical verification through experiments on various datasets.


## Implementation on operator learning, age estimation, and brain-age estimation

Implementations for the three tasks (a synthetic dataset, and two real-world datasets (AgeDB-DIR, UTKFace)) are provided in the separate folders. 


## Implementation

### Datasets and pre-trained models
We have employed four datasets to verigy our model and alongside sota methods, including IXI, AgeDB-DIR, UTKFace, and BVCC dataset.

The IXI dataset can be downloaded at https://brain-development.org/ixi-dataset/;

The AgeDB-DIR dataset can be downloaded at https://ibug.doc.ic.ac.uk/resources/agedb/;

The UTKFace dataset can be downloaded at https://susanqq.github.io/UTKFace/;

The BVCC dataset can be downloaded at https://zenodo.org/records/6572573.

The previous required files with CLSS models are shared at [CLSS models](https://hkustconnect-my.sharepoint.com/:f:/g/personal/wdaiaj_connect_ust_hk/Eu_ZWAv3ZCNHvNl4U24F-7sBnr9Ur57IWtbBHTnyIOGmdQ?e=VRNVGb)

The required files with GCLSS models are also shared at [GCLSS]

Links are also available in the folders for the individual tasks.

### Usage

- Environment
- Train and Evaluation
  
- Results



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
