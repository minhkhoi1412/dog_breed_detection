# Dog Breed Classification

## Content:
- [Project Description](#project-description)
- [Files](#files)
- [Installation Instruction](#installation-instruction)
- [Usage Instruction](#usage-instruction)

Note: If you want to try out the Notebook, please download data as instructed in the notebook. There is code to download and use the data. 

The Notebook contains the [REPORT](./notebook/dog_app.ipynb) required for this project.

## Project Description
This is the Capstone project for the Udacity Data Science Nanodegree course.
For the packages used in the project, please take a look at [requirements.txt file](requirements.txt)

### Project Overview
This project will accept any user-supplied image as input. If a dog is detected within the image, it'll provide an estimate of the dog's breed. If somebody's is detected, it'll provide an estimate of the person that's most resembling.

### Problem Statement
Each major breed has their own sub-breeds and identifying features, like color of fur, texture, head shape, ear shape, and face expression. it's even harder if we are able to only see images of dogs.
My main objective is to train and evaluate a CNN model that takes a RGB image of a dog as an input and predict the breed with the very best accuracy possible. I will be able to simply evaluate some models (VGG16, ResNet50, my own model) and gain a basic knowledge of Computer Vision in Data Science.

### Metrics
Since this is a classification problem, evaluation is based on accuracy. Basically, the model is evaluated on the test set and in percentage.


## Files
```
Dog_breed_detection
│   README.md
│   requirements.txt // required packages
|	LICENSE.txt
│
└───app
│   │   dog_app.py
|	|	run.py
│   │
│   └───upload_folder // contains images uploaded from users
│   │
│   └───templates // contains HTML template files for the webapp.
│       │   index.html
│   
└───bottleneck_features
│   │   DogVGG16Data.npz
│   │   DogResnet50Data.npz
│
└───haarcascades
│   │   haarcascade_frontalface_alt.xml // model to detect human faces
│
└───notebook
|   |   images
│   │   dog_app.ipynb
|	|	download_workspace.ipynb
|	|	extract_bottleneck_features.py
|
└───saved_models //contains saved data for dog breed identification model developed in the notebook
│   │   weights.best.from_scratch.hdf5
|	|	weights.best.Resnet50.hdf5
|	|	weights.best.VGG16.hdf5
│
```

## Installation Instruction
Once you have pull the repo, or download and unzip this repo, please follow the step below to complete the setup.

- Setup an environment (>= Python 3.6) and install packages from (requirements.txt)[requirements.txt].

## Usage Instruction
- After installing the packages, simply run `python run.py` and point you browser to http://127.0.0.1:8080/. The app itself is self-explanatory.
