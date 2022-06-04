# ldct-denoising

## Table of content
* [Overview](https://github.com/kevinmfreire/ldct-denoising#overview)
* [Background and Motivation](https://github.com/kevinmfreire/ldct-denoising#background-and-motivation)
* [Objective](https://github.com/kevinmfreire/ldct-denoising#objective)
* [Dataset](https://github.com/kevinmfreire/ldct-denoising#dataset)
* [Practical Applications](https://github.com/kevinmfreire/ldct-denoising#practical-applications)
* [Usage](https://github.com/kevinmfreire/ldct-denoising#usage)
* [References](https://github.com/kevinmfreire/ldct-denoising#references)

## Overview
Medical Imaging has been a growing topic in computer vision for its life saving applications.  Medical professionals rely on good quality images of their medical scans in order to correctly identify tumours or other anomalies.  Recent studies have shown that Computerize Tomography CT scans have great results using high radiations for their X-Rays.  Studies have shown that these (CT) Scans have produced more than half the radiation from medical use which results in problems for long term use of these expensive machines.  Some solutions have involved reducing the X-Ray magnitude in order to reduce exposure to X-Ray but this results in lower quality images with a lot of noise.  We implemented a denoising neural network trained on medical images to reduce the noise from low dose CT Scans.  This model is capable of producing high dose quality imaging using low dose CT Scans at a reasonable rate.

## Background and Motivation
Computerize Tomography (CT) has enabled direct imaging on the 3-dimensional structure of different organs and tissues inside the human body in a non-invasive manner. CT scans are constructed by combining the X-ray scans taken on several angles and orientations.  It has several utilities but very useful in detecting lesions, tensions, tumours, and metastasis.  It can reveal their presence and the spatial location, size and extent of the tumour.  CT imaging has become a frequent tool for cancer diagnosis, angiography, and detecting internal injuries.  However, despite the evidence of its utility for diagnosis and patient management, the potential risk of radiation-induced malignancy exists.  Studies found that CT alone contributes to almost half of the total radiation exposure from medical use alone.  Recent studies reveal that as much as 1.2 - 2\% of cancers may eventually be caused by the radiation dose conceived by the patient while undergoing CT scans.  To reduce the risk, the principles of ALARA (As Low As Reasonably Achievable) is now a profound practice predicted in CT imaging.  In regards to this practice, Low Dose CT (LDCT) is a promising solution in reducing radiation exposure.  In low dose CT, radiation exposure is decreased by lowering the tube current, or voltage.  However, by reducing the tube voltage or current it introduces several artifacts and lowers the diagnostic quality of the LDCT image.  In order to boost the quality of an LDCT image, the reconstruction of LDCT has become a primary research.  There are various methods that can be classified under three categories: (a) iterative reconstruction, (b) sinogram filtration based techniques, and (c) image post-processing based technique.  In recent times, researchers were trying to develop new iterative algorithms (IR) for LDCT image reconstruction.  Iterative reconstruction algorithms considerably suppresses the image noise, but still lose some details and suffer from remaining artifacts.  Other disadvantages with IR techniques is the high computational cost, which is a bottle neck in practical utilization.  Sinogram filtration on the other hand, directly works on the projection data before reconstructing the image and is more computationally economical than the IR technique.  However, the data of commercial scanners are not readily available to users, and this method suffers from edge blurring and the resolution loss.  Many efforts were made in the image domain to reduce LDCT noise and suppress artifacts.

With the explosive evolution of deep neural networks, the LDCT denoising task is now dominated by deep neural networks.  However, the research on deep learning-based LDCT denoising is confined to designing a network architecture based on the vanilla convolution operation.  There has been a surge in interest in designing General Adversarial Networks (GAN), mainly for image generation and becoming popular in denoising medical images.  In many image-related reconstruction tasks, it is known that minimizing the per-pixel loss between the output image and the ground truth alone generate either blurring or makes the result visually unappealing \cite{huang2017beyond}.  The same effect was observed in the traditional neural network-based CT denoising works.  The adversarial loss introduced by GAN can be treated as a driving force to push the generated image to look as close to as the groung truth image or in this case the Normal Dose CT image (NDCT) which also reduces the blurring effect. Further more, an additional perceptual loss was also introduced to measure the feature of the denoised image, with a focus on areas that the human eye cannot see.

## Objective
The overall objective of this project are as follows:
    1) Develop a Residual-Inception Encoder Decoder Generative Adversarial Network (RIED-GAN)
    2) Train the model using a perceptual loss function using pre-trained VGG16 model, a Dissimilarity structural Index measurement loss function (dsim), and the GAN loss function.
    3) Test network using an LDCT from 1 patient by measuring both the SSIM, RMSE and Peak-Signal-to-Noise Ratio (PSNR) between the noisy, generated and ground truth images.

## Dataset
All data was obtained through the [Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758026) website that I picked out myself.  There are a total of 10 patients, and it's about 2GB of data.

## Practical Applications
Thsi model can be used for removing noise and restoring the image as close to as a Normal Dose CT scan.  It has not been tested on other noisy images as it was only trained on LDCT images, if you wish to train the model on other datasets feel free.

## Usage
* To get started you must clone the repository:
```
git clone https://github.com/kevinmfreire/ldct-denoising.git
```

* Set up a virtual environment.
```
virtualenv .virtualenv/ldct-denoising
```

* Download all necessary requirments:
``` 
pip install -r requirements.txt
```
Warning: If you experience version issue with backpropagation then downgrade your torch version to 1.4.0 (it is commented in the requirements.txt).

* Preprocess your data by running the command below and changing the path within the script (you have the option to normalize data within the prep.py script):
```
prep.py
```

* Then to train and test your data you can run both scrips below however you would want to make a few changes in the trainign algorithm if you'd like:
```
python train.py
python test.py
```

## References