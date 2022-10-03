# ReLoc: A Restoration-Assisted Framework for Robust Image Tampering Localization

# Overview
This is the implementation of the method proposed in "ReLoc: A Restoration-Assisted Framework for Robust Image Tampering Localization" with Pytorch(1.9.0 + cu102). The aim of this repository is to achieve robust image tampering localization.
# Network Architecture!
![image](https://github.com/ZhuangPeiyu/ReLoc/blob/main/models/ReLoc.png)
# Files structure of Dense-FCN-for-tampering-localization
- main_train.py
- metrics.py
- models
  - denseFCN.py
  - SCUnet.py
- checkpoint


# How to run
## Train the model from scratch
python main_train.py


# Acknowledgments
The tampering localization methods and restoration method used in this paper can find in the following links:
- SCSE-Unet: https://github.com/HighwayWu/ImageForensicsOSN
- MVSS-net: https://github.com/dong03/MVSS-Net
- DenseFCN: https://github.com/ZhuangPeiyu/Dense-FCN-for-tampering-localization
- SC-Unet: https://github.com/cszn/SCUNet/
