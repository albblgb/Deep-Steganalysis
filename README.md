# Deep-Steganalysis

#### List of the reproduced papers
- XuNet (SPL2016): [**Structural Design of Convolutional Neural Networks for Steganalysis.**](https://ieeexplore.ieee.org/abstract/document/7444146) 
- YeNet (TIFS2017): [**Deep Learning Hierarchical Representations for Image Steganalysis.**](https://ieeexplore.ieee.org/abstract/document/7937836)
- StegNet (IH&MMSec2017): [**Fast and Effective Global Covariance Pooling Network for Image Steganalysis.**](https://dl.acm.org/doi/abs/10.1145/3335203.3335739)
- SRNet (TIFS2019): [**Deep Residual Network for Steganalysis of Digital Images.**](https://ieeexplore.ieee.org/abstract/document/8470101)
- ZhuNet (TIFS2020): [**Depth-Wise Separable Convolutions and Multi-Level Pooling for an Efficient Spatial CNN-Based Steganalysis.**](https://ieeexplore.ieee.org/abstract/document/8809687)
- SiaStegNet (TIFS2021): [**A Siamese CNN for Image Steganalysis.**](https://ieeexplore.ieee.org/document/9153041)

## Dependencies and Installation
- Python 3.8.13, PyTorch = 1.11.0
- Run the following commands in your terminal:

  `conda env create -f env.yaml`  

  `conda activate pyt_env`


## Get Started
#### Training
1. Change the code in `config.py`

    `line4: mode = 'train'`
   
    `line17: train_data_dir = ''`
   
    `line18: val_data_dir = ''`

    `line20: stego_img_height = `
   
    `line21: stego_img_channel = `

3. Run `python *net.py`. For example, `python srnet.py`

#### Testing
1. Change the code in `config.py`

    `line4: mode = 'test' `

    `line19: test_data_dir = ''`
  
    `line36-41: test_*net_path = ''`

3. Run `python *net.py`

- The trained steganalysis networks will be saved in 'checkpoint/'
- The results and running logs will be saved in 'results/'
 
## Others
- If you find our code useful for your research, please give us a star.
- We don't adopt the default settings from the literature. Instead, all stegeanalysis networks are optimized using Adam slover with a weight decay of 1e-5 and an initial learning rate of 2e-4.
