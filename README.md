# Dog-Breed-Classifier:
CNN Project [Udacity Deep Learning Nanodegree]


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JyWFFhU0nWacbkleapXcBpp89O62eAsR?usp=sharing)]

## Project Overview:

This repo contains all my work for Project 1 of Udacity's Neural Network Foundation Nanodegree Program. In this project, I have learned how to build a pipeline to process real-world, user-supplied images. Given an image of a dog, my algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.
                       Along with exploring state-of-the-art CNN models for classification, I have made important design decisions about the user experience for our app. By completing this lab, I understood the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline. Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer.

**Udacity's original repo is [here](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-dog-classification)**


## Data downloadable on:
https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip

https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip
(no need to download , its taken care in the notebook)


## Disclaimer:

As a CS minor student of IIT Kharagpur and a long-time self-taught learner, I have completed many CS related MOOCs on Coursera, Udacity, Udemy, and Edx. I do understand the hard time you spend on understanding new concepts and debugging your program. Here I released these solutions, which are **only for your reference purpose**. It may help you to save some time. And I hope you don't copy any part of the code (the programming assignments are fairly easy if you read the instructions carefully), see the solutions before you start your own adventure. This Project is almost one of the simplest Machine Learning Project I have ever taken, but the simplicity is based on the fabulous course content and structure. It's a treasure given by Udacity team.



### Conclusion
This project introduced me to what a Convolutional Nerual Network is, how its architecture is made with filters, pooling and dropout layers and the benefits that can be gained when taking advantage of transfer learning.
<br><br>
To view the full project [click here](https://nbviewer.jupyter.org/github/MrDaubinet/Dog-Breed-Classifier/blob/master/dog_app.ipynb)

# Get started
```
https://github.com/rahulgarg28071998/Dog-Classifier-Udacity-nanodegree
jupyter notebook
```
# Requirements
Jupyter notebook must be installed.
Python must be installed. The following python modules must be installed.
```
opencv-python
jupyter
matplotlib
pandas
numpy
pillow
scipy
tqdm
scikit-learn
scikit-image
seaborn
h5py
ipykernel
bokeh
pickleshare
```
# Chalenges
## Issues with reproducibility
I started with using the Jupyther notebook supplied as a part of the lesson. But! Running the same code was producing completely different results. I tried restarting the kernel,but it didn't help.  
To have more flexibilty, I switched to a notebook ran on AWS where I could also restart the notebook and even reboot the machine. Still, running the same code was producing diffferent results. One time it would get stuck at validation loss around 4.8 for thr first 10 epochs (and then I would shut it down).   
Another time, using the same code, it would learn very efficiently and obtain validation loss below 4.0 by 10th epoch. 

## Problems with CUDA on AWS AMIs

Udacity's lesson about [Cloud Computing](Deep Learning AMI with Source Code) recommends using the "Deep Learning AMI with Source Code (CUDA 8, Ubuntu)" AMI run notebooks.   
CUDA 8 is old, and AMI for CUDA 9 is available. 

## Main Kaggle Competition link :
Dog Breed Identification (https://www.kaggle.com/c/dog-breed-identification)
