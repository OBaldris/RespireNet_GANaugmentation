
# DATA AUGMENTATION WITH GAN  FOR CLASSIFICATION IMPROVEMENT

Final project for CPIA class, created by Marc Molla, Miquel Sánchez and Oriol Baldrís




## Dependencies:

To run this project, you will need to add the following environment variables to your .env file

` Python3`
`Pytorch (torch, torchvision and other dependencies for Pytorch)`
` Numpy`
`Librosa (0.8.0)`
`nlpaug (0.0.14)`
`OpenCV (4.2.0)`
`Pandas (0.22.0)`
`scikit-learn (0.23.1)`
`tqdm (4.48.0)`
`cudnn (CUDA for training on GPU)`

If working on a virtual environment, install all. If working from an online editor line Colab, check which are preinstalled




## Data

This project uses the ICBHI dataset. Make sure to download from 

https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database
## Preprocessing
To run the preprocessing of audio, run file preprocessing.py
Inside file, just go to second and third to last line and change directory to read audio data and change:

input_folder_path 
output_folder_path 

To desired directories

Also, in line, 51, changes values mark1 and mark2 to desired. This value represent which kind of cycle is saved. 
Example:
mark1= 0 mark2= 1  (this would mean save audio segments with crackles)

To make it easier to organize data, create four folders named 00, 01, 10, 11 for each data group. 
## GAN
CAREFUL!!!
Do not load images at the same time with different marks (different kind 00, 01, 10, 11)

To run GAN, change notebook variables:

    directory_path 
    image_path
    weights_path


If running from colab, make sure all folders are properly created. directory_path is the file with the input images. Make sure to clearly differentiate between 00, 01, 10 and 11 both for input and for output directories.


## RespireNet
In order to run the training loop (not necessary if only willing to generate new data) download repository from https://github.com/microsoft/RespireNet and follow instructions. If willing to run RespireNet programatically, use file resnet.ipynb and change directories to suit your environment. This model needs to run on parallel GPUs, such as the GPU T4 x2 from the kaggle or colab platform.
## Authors

- [@OBaldris](https://github.com/OBaldris)

- [@mollamarc](https://github.com/mollamarc)

    Miquel Sánchez

