# Deep-Steganalysis

#### List of reproduced papers
- XuNet: [**Hiding images within images.**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8654686) 
- YeNet: [**HiDDeN: Hiding Data With Deep Networks.**](https://openaccess.thecvf.com/content_ECCV_2018/html/Jiren_Zhu_HiDDeN_Hiding_Data_ECCV_2018_paper.html)
- ZhuNet: [**High-Capacity Convolutional Video Steganography with Temporal Residual Modeling.**](https://dl.acm.org/doi/abs/10.1145/3323873.3325011)
- SiaStegNet: [**HiNet: Deep Image Hiding by Invertible Network.**](https://openaccess.thecvf.com/content/ICCV2021/html/Jing_HiNet_Deep_Image_Hiding_by_Invertible_Network_ICCV_2021_paper.html)
- SRNet: [**HiNet: Deep Image Hiding by Invertible Network.**](https://openaccess.thecvf.com/content/ICCV2021/html/Jing_HiNet_Deep_Image_Hiding_by_Invertible_Network_ICCV_2021_paper.html)
- StegNet: [**HiNet: Deep Image Hiding by Invertible Network.**](https://openaccess.thecvf.com/content/ICCV2021/html/Jing_HiNet_Deep_Image_Hiding_by_Invertible_Network_ICCV_2021_paper.html)

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

