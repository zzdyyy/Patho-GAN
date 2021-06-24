# Patho-GAN
This is project for the paper "Explainable Diabetic Retinopathy Detection and Retinal Image Generation".

Patho-GAN can generate diabetic retinopathy(DR) fundus given Pathological descriptors, vessel segmentation and a noise vector. 


# Testing
To run the test code, please check xxx.

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
4. Compute Activation Map A0(x): download data/detector.h5, and run in Patho-GAN's root directory:
    ```
    python tfpipe_dump_activation.py "data/IDRiD/train_512/*.jpg" --dump_to IDRiD_train_dump --visualize # 54 images
    python tfpipe_dump_activation.py "data/IDRiD/test_512/*.jpg" --dump_to IDRiD_test_dump --visualize # 27 images
    ```
    for other datasets, run like this:
    ```
    python tfpipe_dump_activation.py "data/retinal-lesions/resized_512/3*.jpg" --dump_to retinal-lesions_train_dump --visualize # 337 images
    python tfpipe_dump_activation.py "data/retinal-lesions/resized_512/[124-9]*.jpg" --dump_to retinal-lesions_test_dump --visualize # 1256 images
    
    python tfpipe_dump_activation.py "data/FGADR/resized_512/*.jpg" --dump_to FGADR_dump --visualize # 1842 images
    ```
5. Generate the numpy of the dataset. Run `python to_npy.py` in `data/IDRiD/`.
6. Generate descriptors for test samples. Download data/imagenet-vgg-verydeep-19.mat, run in Patho-GAN's root directory:
    ```
    python DMB_build_test_samples.py IDRiD IDRiD_55.jpg IDRiD_61.jpg IDRiD_73.jpg IDRiD_81.jpg
    python DMB_build_test_samples.py retinal-lesions 250_right.jpg 2016_right.jpg 2044_left.jpg 2767_left.jpg
    python DMB_build_test_samples.py FGADR 0508.jpg 0549.jpg 0515.jpg 0529.jpg
    ```
7. Start training:
    ```
    python Train.py IDRiD IDRiD_55.jpg IDRiD_61.jpg IDRiD_73.jpg IDRiD_81.jpg
    python Train.py retinal-lesions 250_right.jpg 2016_right.jpg 2044_left.jpg 2767_left.jpg
    python Train.py FGADR 0508.jpg 0549.jpg 0515.jpg 0529.jpg
    ```
