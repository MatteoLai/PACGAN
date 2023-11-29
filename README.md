[![DOI](https://zenodo.org/badge/651442582.svg)](https://zenodo.org/badge/latestdoi/651442582) [![Open in Code Ocean](https://codeocean.com/codeocean-assets/badge/open-in-code-ocean.svg)](https://codeocean.com/capsule/6317229/tree) [<img src="https://img.shields.io/badge/  -Dockerhub-blue.svg?logo=docker&logoColor=white">](<https://hub.docker.com/r/aiformedresearch/pacgan>) 

# PACGAN

This repository contains the implementation of PACGAN (Progressive Auxiliary Classifier Generative Adversarial Network), a model inspired by the architecture of the [ACGAN](https://arxiv.org/abs/1610.09585) and by the training procedure of the [Progressive Growing GAN](https://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of/karras2018iclr-paper.pdf), which was designed and implemented by the [AI for Medicine Research Group](https://aiformedresearch.github.io/aiformedresearch/) at the University of Bologna. In this study, we applied this framework for the generation of synthetic full-size brain MRI images of Alzheimer's patients and healthy control and to perform classification between the two classes. 

<p align="center">
  <img src="Training/PACGAN.JPG" width="1000" title="PACGAN">
</p>

## Index
1. [License](#license)
2. [Preprocessing](#preprocessing)
3. [Installation](#installation)
4. [Training](#training)
5. [Inference](#inference)
6. [Set the configuration file](#set-the-configuration-file)
7. [Citation](#citation)

## License
Please read the [LICENSE.md](./LICENSE.md) file before using it.

## Preprocessing
The PACGAN model has been trained on T1-weighted brain MRI of Alzheimer's patients and healthy controls, obtained from [ADNI](https://adni.loni.usc.edu/). Each volume was co-registered to the MNI152 standard template space (at 1 mm voxel size), available in the FSL package (version 6.0), through a linear registration with 9 degrees of freedom and trilinear interpolation using FSL's FLIRT. Each output of the co-registration was then made cubic, with a size of 256 $\times$ 256 $\times$ 256. Finally, we extracted the central axial slice (corresponding to slice number 127) and concatenated all the axial slices into a single NIfTI volume with a size of 256 $\times$ 256 $\times$ \#images.

Patients and controls were matched for age and sex through the script [ADvsHC_matching.py](Preprocessing/ADvsHC_matching.py).  

Subsequently, the matched subjects were divided into two sets using a stratified holdout approach. This partitioning, implemented in [divide_TrainTest.py](Preprocessing/divide_TrainTest.py), allocated 80% of the images for training and validation, and the remaining 20% for testing. Particular attention was given to ensuring that the age and sex distributions were matched between the sets, while also guaranteeing that images originating from the same subject were placed in separate subsets. The script [main.py](Training/main.py) implements the possibility to futrther divide the isolated 80% of the images into training and validation sets, all the while ensuring that images belonging to the same subject are allocated to different subsets.

To prepare the data for training, the script [divide_TrainTest.py](Preprocessing/divide_TrainTest.py) requires the following inputs:
- the NIfTI file containing all the slices of the matched subjects (*path_img*);
- the list of the paths to the volumes from which each image was extracted (*path_list*);
- the labels corresponding to each subject, downloaded from the [ADNI](https://adni.loni.usc.edu/data-samples/access-data/) site (*path_csv*).

and will return the **data ready for training the PACGAN model**:
- two NIfTI files for each resolution needed for the training of PACGAN (256 $\times$ 256, 128 $\times$ 128, 64 $\times$ 64, 32 $\times$ 32, 16 $\times$ 16, 8 $\times$ 8 and 4 $\times$ 4), one for the training and one for the test (saved in two separated folders);

  Below is an example of the folder hierarchy, which can be customized using the [config.json](Training/config.json) file (refer to the [Set the configuration file](#set-the-configuration-file) section):
  ```
  data/ADNI
        └─── ADNI_train
        |        └─── Real256x256.nii.gz
        |        └─── Real128x128.nii.gz
        |        └───   ...
        |        └─── Real4x4.nii.gz
        |        └─── labels.csv
        └─── ADNI_test
                 └─── Real256x256.nii.gz
                 └─── Real128x128.nii.gz
                 └───   ...
                 └─── Real4x4.nii.gz
                 └─── labels.csv
  ```

- two csv files (one for training and one for testing) with three columns: 
  - '*ID*', the index of the image in the corresponding NIfTI file (0,1, ..., #images); 
  - '*Label*', the label of the image (0 for CN, 1 for AD);
  - '*Subject_ID*', the ID of the subjects. This information is crucial during the division into training and validation sets to ensure that images from the same subject are kept separate, thus preventing any data leakage issues.
 
  | ID | Label | Subject_ID |
  |----|-------|------------|
  | 0  |   0   |   S_0001   |
  | 1  |   0   |   S_0002   |
  | 2  |   1   |   S_0003   |
  | 3  |   1   |   S_0003   |
  |... |  ...  |    ...     |
  |#images |  0  |  S_000N   |

## Installation

### Installation with Anaconda
#### 1. Clone the repository
   
Open the terminal or command prompt on your local machine and navigate to the directory where you want to clone the repository
```
cd /path_to/PACGAN_repo
```

To clone the repository, run the following command
```
git clone https://github.com/aiformedresearch/PACGAN.git
```

Finally, navigate inside the `PACGAN` repository you just cloned:
```
cd PACGAN
```

Before to proceeding, please note that [CUDA installation](https://developer.nvidia.com/cuda-downloads) is required. If CUDA is not compatible with your device, you can still run the code on CPU. However, please note that the results may vary compared to running on CUDA-enabled devices.

#### 2. Create the conda environment
Install [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) choosing the appropriate version based on your operating system.

Open the Anaconda Prompt and create the environment with Python version 3.8.8:
```
conda create --name PACGAN_env python=3.8.8
```

Activate the newly created environment:
```
conda activate PACGAN_env
```

Add the [conda-forge](https://conda-forge.org/) channel to access additional packages required for this project:
```
conda config --env --add channels conda-forge
```

To expedite the installation process, we recommend installing the required packages in the following order:

i) Install the [Pytorch](https://pytorch.org/) packages:
```
conda install pytorch==1.9.0 torchvision==0.10.0 torchmetrics==0.11.0 -c pytorch -c nvidia
```

ii) install the [torch-fidelity](https://pypi.org/project/torch-fidelity/) package separately:
```
conda install torch-fidelity==0.3.0
```

iii) install the remaining necessary packages listed in the [requirements.txt](requirements.txt) file:
```
conda install --file requirements.txt
```

Once you have completed these steps, you can proceed to the [Training](#training) or [Inference](#inference) section.

### Installation with Docker
Install [Docker](https://docs.docker.com/get-docker/) selecting the proper operating system.

Once Docker is installed, pull the docker image
```
docker pull aiformedresearch/pacgan
```

Run the Docker container
```
docker run -it -v /absolute_path/to/data_folder:/PACGAN/data --gpus all aiformedresearch/pacgan
```
This command will start a new container using the [aiformedresearch/pacgan](https://hub.docker.com/r/aiformedresearch/pacgan) image, with GPU support enabled (`--gpus all`). Please note that you can also select a specific GPU to use, e.g., `--gpus 0`. If CUDA is not available in your device, remove the `--gpus` flag to train on CPU.

The `-v` flag is used to mount the directory containing your training data to the `/PACGAN/data` directory inside the container. 
Make sure to replace `/absolute_path/to/data_folder` with the absolute path to the directory containing data to be used for the training of PACGAN. 
Please note that the *data* folder must be organised as explained in the [Preprocessing](#preprocessing) section.

Once the container is running, you can proceed with the steps outlined in the [Training](#training) section. Type `exit` when you want to exit from the Docker container.

## Training
To train the model on new data, you can change the path to the data and the hyperparameters of the model by modifying the [config.json](Training/config.json) file, as detailed in the [Set the configuration file](#set-the-configuration-file) section.

After this, you can run the training code:
```
python Training/main.py -j Training/config.json
```
Considering that the training will take a few hours, we suggest launching the process in the background and drawing up the output in an `output.log` file by running:
```
python Training/main.py -j Training/config.json > output.log 2>&1 &
```

## Inference
The trained PACGAN model has dual functionality, serving both image generation and classification purposes. Throughout the training process, the generator and discriminator models, configured to utilize the *DEVICE* set in [config.json](Training/config.json),  are consistently saved on the CPU upon completion. This ensures their compatibility across all devices, including those without CUDA support, facilitating their restoration and utilization.

Unfortunately, due to the file size limit imposed by GitHub, we were unable to upload the pre-trained generator and discriminator models to this repository. However, you can conveniently download them from the *data* folder of the [Code Ocean Capsule](https://codeocean.com/capsule/6317229/tree).

### Generate images
The **Generator** can generate synthetic images by calling the function [Generator.py](Training/Generator.py), which loads the weights of the model trained at a specific resolution exploiting the generator model defined in [models.py](Training/models.py). You can set the size and the number of images to generate. Here there is an example of calling it from the command line to generate 100 images for each class (`n_images_xCLASS`=[100,100]) with the size 256 $\times$ 256 (`img_size`=256):
```
python Training/Generator.py \
  --img_size 256 \
  --n_images_xCLASS 100 100 \
  --model_path /path/to/generator_model.pt \
  --images_path /path/where/save/generated_images.nii.gz \
  -j Training/config.json \
  --device cuda \
  --gpus 0 1
```
The images generated by the generator saved in `model_path` will be saved in a NIfTI file at `images_path`. The output images are equally divided between the three classes. 
If, for example, CLASS_SIZE=2 (the same used for the training, indicated in [config.py](Training/config.py)), the first half of the images will belong to class 0, and the second half to class 1. Note that the inputs `--device` and `--gpus` are optional; by default, the model is loaded on the CPU.


### Discriminate new istances

The **Discriminator** can classify a batch of new images by calling the function [Discriminator.py](Training/Discriminator.py). It can be exploited to do inference on new images organized in a nifti file and will return a csv with the predictions by running from the command line:
```
python Training/Discriminator.py \
  --img_path /path/to/nifti_images_to_classify.nii.gz \
  --model_path /path/to/discriminator_model.pt \
  -j Training/config.json \
  --device cuda \
  --gpus 0 1
```
The prediction will be performed on the NIfTI file saved in `img_path` exploiting the discriminator stored in `model_path`.
This function will return the predictions in the file *predictions.csv*, saved in `img_path`. Note that the inputs `--device` and `--gpus` are optional; by default, the model is loaded on the CPU.

## Set the configuration file
The PACGAN model was developed to facilitate training on datasets that may contain various types of images, such as grayscale or RGB, with varying dimensions that may deviate from the default size of 256. To allow for customization of the model's architecture, hyperparameters, and configuration settings, the [config.json](Training/config.json) file can be modified. The following are the key components available for customization within the config.py file.

The **folders** containing the data:
- *DATA_TRAIN_DIRECTORY* and *LABEL_TRAIN_DIRECTORY* - paths to the data and corresponding labels used for the training and validation sets. The division of the data is performed in the [main.py](Training/main.py) script through a stratified holdout approach. Data are supposed to be organized in a NIfTI file with dimension [W, H, #channels, #images]. The labels are expected to be provided in a csv file with three columns as described above;
- *DATA_TEST_DIRECTORY* and *LABEL_TEST_DIRECTORY* - paths to the data and corresponding labels of the test set. The data should be organized in a NIfTI file with dimension [W, H, #channels, #images]. The labels should be provided in a csv file with three columns as described earlier;
- *IMAGE_DIRECTORY* and *MODEL_DIRECTORY* - paths to the directories where the outputs of the algorithm will be saved;

The characteristics of the **input images**:
- *IMAGE_CHANNELS* - the number of image channels, set 1 for grayscale images and 3 for RGB images;
- *CLASS_SIZE* - the number of class labels in the dataset;

The charachteristics of the **models architectures**:
- *IN_CHANNELS* and *FACTORS* - the number of channels for the CNNs of the generator and discriminator models, along with the factors used for scaling in the progressive growing structure (default IN_CHANNELS=512 and FACTORS=[1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]);
- *PROGRESSIVE_EPOCHS* - the number of epochs to train at each resolution level during the progressive growing process (default PROGRESSIVE_EPOCHS=100\*len(FACTORS)=[100, 100, 100, 100, 100, 100, 100];

The **hyperparameters** of the model:
- *LEARNING_RATE* - learning rate of the model;
- *BATCH_SIZES* - batch size used at each level of the training (default BATCH_SIZES=[32, 32, 32, 16, 16, 16, 16]);
- *EMBEDDING_DIM* - the dimension of the output of the class embedding (default EMBEDDING_DIM=3);
- *Z_DIM* - the dimension of the random vector, which serves as input of the generator (default Z_DIM=512);
- *CRITIC_ITERATIONS* - the number of iterations the critic is trained at each epoch (which can vary at different resolutions - default CRITIC_ITERATIONS=[1, 1, 1, 1, 1, 2, 2]);
- *LAMBDA_GP* - the weight of the gradient penalty term in the loss function (default $\lambda_{GP}=10$);
- *LAMBDA_CLS1* - the weight of the cross entropy loss between the real labels and the labels estimated by the discriminator for the *generated* images (default $\lambda_{cls1}=4$);
- *LAMBDA_CLS2* - the weight of the cross entropy loss between the real labels and the ones estimated by the discriminator for the *real* images (default $\lambda_{cls2}=\lambda_{cls1}/2$).

The **device** to use for the training:
- *DEVICE* - the device to be used for training (either [CUDA](https://developer.nvidia.com/cuda-downloads) or CPU). By default, if cuda is avalaible, it will be utilized. Please note that using CPU may result in variations in the obtained results compared to using CUDA.
- *GPUS_N* - the index of GPU to be used for parallelizing the training, if CUDA is available. If you have only one GPU, set GPUS_N=0;

The **training modality**:
- *VALIDATE* - set True if you want to validate the hyperparameters on the validation set; if *VALIDATE*=True, the *best_epoch* for each step of the training will be determined by evaluating the validation loss from the *START_CHECK_AT* % of the *PROGRESSIVE_EPOCHS* (default *START_CHECK_AT*=60% of the number of training epochs);
- *TESTING* - when set to True (and VALIDATE=*False*), the model will be trained on the entire set of data saved in *DATA_TRAIN_DIRECTORY* (training+validation), and the performance of the final model will be evaluated on the test set. By default, the number of epochs for each step will be the *best_epoch* found in the validation phase; you can change this (setting a fixed number of training epochs) by setting *TEST_USING_BEST_MODEL*=False.

The **number of images to generate** at the end of the training:
- *N_IMAGES_GENERATED* - by setting *GENERATE*=True, the algorithm will generate *N_IMAGES_GENERATED* at the end of the training. This  allows for the computation of the quantitative metrics ([FID](https://torchmetrics.readthedocs.io/en/stable/image/frechet_inception_distance.html), [KID](https://torchmetrics.readthedocs.io/en/v0.8.2/image/kernel_inception_distance.html) and [SSIM](https://scikit-image.org/docs/stable/auto_examples/transform/plot_ssim.html)) to evaluate the realness of the synthetic images. The synthetic images will be saved in *IMAGE_DIRECTORY*, and the corrisponding metrics in a *labels_syn.csv* file. The values of the metrics are saved in a *metrics.txt* file.  
  - To generate a specific number of images, e.g., 1000 images, set *N_IMAGES_GENERATED*=[1000]. This will generate an equal number of images for each class.
  - To obtain an unbalanced synthetic dataset, set the number of images to generate for each class directly. For example, *N_IMAGES_GENERATED*=[200, 800], in the case of *CLASS_SIZE*=2.

To **re-train the model**:

After the training of the model, you have the option to re-train it starting from a specific resolution *START_TRAIN_AT_IMG_SIZE* by setting *LOAD_MODEL*=True.

## Citation

If you utilize this package, kindly cite it using the provided DOI ([10.5281/zenodo.8021009](https://doi.org/10.5281/zenodo.8021009)) to find the latest citation on Zenodo. Various citation formats can be found in the **Export** section located at the bottom of the page.
