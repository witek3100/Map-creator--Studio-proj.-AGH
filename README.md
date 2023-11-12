# Overview
## Problem Statement
Purpose of the project is to create deep learning based system for creating maps of urbanized areas from satellite images.
## Approach
The concept of the entire project is quite simple, we take a satellite image, divide it into small pieces of 50x50 pixels, which are then classified by a previously trained model, and finally, based on the classified data, we create a map.
## Used technologies
Entire projects is written in python, satellite images are fetched from qgis, ml part is made using PyTorch.

# Project structure
<pre>
   ├── data
   |     ├── processed            # Data for training
   |     |       ├── images
   |     |       ├── test.csv
   |     |       ├── train.csv
   |     |       └── validation.csv
   |     ├── raw                     # Data to label and split
   |     |    └── images
   |     ├── labeler.py            # PyQt gui for labeling images
   |     ├── split_dataset.py        # script spliting data to train, val and test sets
   |     └── split_image.py         # script spliting satellite image to 50x50 pixels tiles
   ├── maps                   
   ├── src
   |     ├── ml
   |     |   ├── dataset            # Pytorch datasets, dataloaders and transformations
   |     |   |      ├── dataset.py
   |     |   |      └── transformations.py
   |     |   ├── models                   # Models definitions and corresponding weights files
   |     |   |  
   |     |   ├── scripts                  # scripts for training, evaluating and predicting
   |     |   |      ├── predict.py
   |     |   |      ├── test.py
   |     |   |      └── train.py
   |     |   └── utils                  # Everything else used for training
   |     |          ├── early_stopping.py
   |     |          ├── metrics.py
   |     |          ├── optimizers.py
   |     |
   |     ├── map_generator             # Code for generating map from model predictions
   |     |         ### TODO ###
   |     ├── qgis_api                  # Api for fetching setallite images
   |               ### TODO ###
</pre>

# Prototype
