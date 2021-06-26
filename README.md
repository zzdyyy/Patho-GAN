# Patho-GAN
This is project for the paper "Explainable Diabetic Retinopathy Detection and Retinal Image Generation".

Patho-GAN can generate diabetic retinopathy(DR) fundus given Pathological descriptors, vessel segmentation and a noise vector. 


# Testing

```bash
pip install -r requirements.txt 

# Download pretrained VGG-19 model
wget -O data/imagenet-vgg-verydeep-19.mat 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

# Download pretrained o_O detector model
gdown -O data/detector.h5 'https://drive.google.com/uc?id=1OI1d3XWM7IyW2igIEq8s-ZyF9vw0vTiw'

# Download IDRiD vessel segmentation, descriptors, and pretrained Patho-GAN model
gdown -O idrid_testing.tar.xz 'https://drive.google.com/uc?id=1Cf1WoaoGf6m7t6z70kpEl1SXOxTeM6Qu'
tar -xvf idrid_testing.tar.xz

# Run test script, and generated `Test/IDRiD_Reconstruct` directory
python Test_reconstruct_DMB.py IDRiD
```

# Training

We take IDRiD dataset for example.

1. Download dataset from [this link](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid), and extract it.
2. To crop and resize images into 512x512, cd into `data/IDRiD`, and run following line :
    ```
    python convert.py --directory '/path_to_extracted_data/IDRiD/A. Segmentation/1. Original Images/a. Training Set' --convert_directory train_512/
    python convert.py --directory '/path_to_extracted_data/IDRiD/A. Segmentation/1. Original Images/b. Testing Set' --convert_directory test_512/
    ```
3. Generate vessel segmentation. Clone modified version of [SA-UNet](https://github.com/zzdyyy/SA-UNet), and run in its root directory:
    ```
    python Test_PathoGAN.py IDRiD
    ```
4. Generate the numpy of the dataset. Run `python to_npy.py` in `data/IDRiD/`.
5. Generate descriptors for test samples. Download data/imagenet-vgg-verydeep-19.mat, run in Patho-GAN's root directory:
    ```
    python DMB_build_test_samples.py IDRiD IDRiD_55.jpg IDRiD_61.jpg IDRiD_73.jpg IDRiD_81.jpg
    python DMB_build_test_samples.py retinal-lesions 250_right.jpg 2016_right.jpg 2044_left.jpg 2767_left.jpg
    python DMB_build_test_samples.py FGADR 0508.jpg 0549.jpg 0515.jpg 0529.jpg
    ```
6. Start training:
    ```
    python Train.py IDRiD IDRiD_55.jpg IDRiD_61.jpg IDRiD_73.jpg IDRiD_81.jpg
    python Train.py retinal-lesions 250_right.jpg 2016_right.jpg 2044_left.jpg 2767_left.jpg
    python Train.py FGADR 0508.jpg 0549.jpg 0515.jpg 0529.jpg
    ```
