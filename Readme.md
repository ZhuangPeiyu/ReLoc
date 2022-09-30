# ReLoc: A Restoration-Assisted Framework for Robust Image Tampering Localization

# Overview
This is the implementation of the method proposed in "ReLoc: A Restoration-Assisted Framework for Robust Image Tampering Localization" with Pytorch( gpu version). The aim of this repository is to achieve image tampering localization.
# Network Architecture
![image](https://github.com/ZhuangPeiyu/Dense-FCN-for-tampering-localization/blob/master/networkArchitecture/158b993b1ea5a0b7ee6e460376e3ce2.png)
# Files structure of Dense-FCN-for-tampering-localization
- Models
- Results
- testedImages
- utilis
- train_demo.py
- denseFCN.py
- test_withoutComputeMetrics.py

# The pre-trained model path
The model trained with Dresden script dataset and fine-tuned with 56 NIST images was uploaded in Dropbox: https://www.dropbox.com/sh/0hkeenrfazob3ci/AAAa6X2hhDnj04LfAR2mSKi9a?dl=0
# How to run
## Test with the trained model

python3 test_withoutComputeMetrics.py

## Train the model from scratch
python3 train_demo.py

# Citation
If you use our code please cite: 

@ARTICLE{9393396,  author={P. {Zhuang} and H. {Li} and S. {Tan} and B. {Li} and J. {Huang}},  
journal={IEEE Transactions on Information Forensics and Security},   
title={Image Tampering Localization Using a Dense Fully Convolutional Network},   
year={2021},  
volume={16},  
number={},  
pages={2986-2999},  
doi={10.1109/TIFS.2021.3070444}}

