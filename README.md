# Deep-Steganalysis

#### List of reproduced papers
- XuNet (SPL2016): [**Structural Design of Convolutional Neural Networks for Steganalysis.**](https://ieeexplore.ieee.org/abstract/document/7444146) 
- YeNet (TIFS2017): [**Deep Learning Hierarchical Representations for Image Steganalysis.**](https://ieeexplore.ieee.org/abstract/document/7937836)
- StegNet (IH&MMSec2017): [**Fast and Effective Global Covariance Pooling Network for Image Steganalysis.**](https://dl.acm.org/doi/abs/10.1145/3335203.3335739)
- SRNet (TIFS2019): [**Deep Residual Network for Steganalysis of Digital Images.**](https://ieeexplore.ieee.org/abstract/document/8470101)
- ZhuNet (TIFS2020): [**Depth-Wise Separable Convolutions and Multi-Level Pooling for an Efficient Spatial CNN-Based Steganalysis.**](https://ieeexplore.ieee.org/abstract/document/8809687)
- SiaStegNet (TIFS2021): [**A Siamese CNN for Image Steganalysis.**](https://ieeexplore.ieee.org/document/9153041)

## Dependencies and Installation
- Python 3.8.13, PyTorch = 1.11.0
- Run the following commands in your terminal:

  `conda env create  -f env.yml`

   `conda activate NIPS`


## Get Started
#### Training
1. Change the code in `config.py`

    `line4:  mode = 'train' ` 

2. Run `python *net.py`, for example, `python wengnet.py`

#### Testing
1. Change the code in `config.py`

    `line4:  mode = 'test' `
  
    `line36-41:  test_*net_path = '' `

2. Run `python *net.py`

- Here we provide [trained models](https://drive.google.com/drive/folders/1lM9ED7uzWYeznXSWKg4mgf7Xc7wjjm8Q?usp=sharing).
- The processed images, such as stego image and recovered secret image, will be saved at 'results/images'
- The training or testing log will be saved at 'results/*.log'


## Dataset
- The models are trained on the [DIV2K](https://opendatalab.com/DIV2K) training dataset, and the mini-batch size is set to 8, with half of the images randomly selected as the cover images and the remaining images as the secret images. 
- The trained models are tested on three test sets, including the DIV2K test dataset, 1000 images randomly selected from the ImageNet test dataset
- Here we provide [test sets](https://drive.google.com/file/d/1NYVWZXe0AjxdI5vuI2gF6_2hwoS1c4y7/view?usp=sharing).

- For train or test on the dataset,  e.g.  DIV2K, change the code in `config.py`:

    `line17:  data_dir = '' `
  
    `data_name_train = 'div2k'`
  
    `data_name_test = 'div2k'`
  
    `line30:  suffix = 'png' `

- Structure of the dataset directory:

<center>
  <img src=https://github.com/albblgb/pusnet/blob/main/utils/dataset_folder_structure.png width=36% />
</center>
 
    
## Others
- The `batch_size` in `config.py` should be at least `2*number of gpus` and it should be divisible by number of gpus.
- The network-generated images are quantified before the evaluation. 

