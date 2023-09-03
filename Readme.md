# ReLoc: A Restoration-Assisted Framework for Robust Image Tampering Localization

# Overview
This is the implementation of the method proposed in "ReLoc: A Restoration-Assisted Framework for Robust Image Tampering Localization" with Pytorch(1.9.0 + cu102). The aim of this repository is to achieve robust image tampering localization.
# Network Architecture!
![image](https://github.com/ZhuangPeiyu/ReLoc/blob/main/models/ReLoc.png)
# Files structure of ReLoc
- codes 
  - models: codes of SCSEUnet <sup>[1]</sup>
  - MVSS_net: codes of MVSSNet <sup>[2]</sup>
  - denseFCN.py: code of DFCN <sup>[3]</sup>
  - SCUNet_main: codes of SCUNet <sup>[4]</sup>
  - metrics.py: code for computing the localization performance.
  - test.py: the testing script.
  - train.py: the training script.
  - configs.py: the config of training ReLoc.
- checkpoints: the weights of ReLoc equipped with 3 localization modules (i.e., DFCN, SCSEUnet, and MVSSNet)
trained on DEFACTO dataset. You can download these files from [Baidu Yun (Code: e5ww)](https://pan.baidu.com/s/1UlQRDXjK6TuhucdOiiyvdQ)


# How to run
## Train the ReLoc model
### 1. Modify the training config of ReLoc in configs.py
### 2. python train.py

## Test the ReLoc model
### 1. python test.py

# Acknowledgments
The tampering localization methods and restoration method used in this paper can find in the following links:
- SCSE-Unet <sup>[1]</sup>: [paper](https://ieeexplore.ieee.org/abstract/document/9686650) and [codes](https://github.com/HighwayWu/ImageForensicsOSN)
- MVSSNet <sup>[2]</sup>: [paper](https://ieeexplore.ieee.org/abstract/document/9789576) and [codes](https://github.com/dong03/MVSS-Net)
- DFCN <sup>[3]</sup>: [paper](https://ieeexplore.ieee.org/abstract/document/9393396) and [codes](https://github.com/ZhuangPeiyu/Dense-FCN-for-tampering-localization)
- SCUnet <sup>[4]</sup>: [paper](https://arxiv.org/abs/2203.13278) and [codes](https://github.com/cszn/SCUNet/)

# Cication
If you use our code please cite:

@ARTICLE{ReLoc,

  title={ReLoc: A Restoration-Assisted Framework for Robust Image Tampering Localization}, 

  author={Zhuang, Peiyu and Li, Haodong and Yang, Rui and Huang, Jiwu},

  journal={IEEE Transactions on Information Forensics and Security}, 

  year={2023},

  volume={18},

  pages={5243-5257}}