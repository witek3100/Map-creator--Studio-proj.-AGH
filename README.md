# Map Creator - Studio-Projektowe-AGH
Purpose of this project is to create maps from sattelite images using deep learning

## Prototype

The prototype creates a map of urbanized areas from a satellite image of Krakow. The entire concept looks as follows: We build a model composed of a pre-trained deep neural network and additional output layers that we train. Then, we divide the initial satellite image into smaller images, classify them, and use this classification to create a map.

### Model
Model base is ResNet34, next we've added three blocks of dense, batch normalization and dropout layers sequence.  
Model definition in model/model.py

### Data
Model is trained on [this dataset](https://github.com/phelber/EuroSAT).  
   See references section

## References

[1] Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. Patrick Helber, Benjamin Bischke, Andreas Dengel, Damian Borth. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.

@article{helber2019eurosat,
  title={Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification},
  author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2019},
  publisher={IEEE}
}

[2] Introducing EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. Patrick Helber, Benjamin Bischke, Andreas Dengel. 2018 IEEE International Geoscience and Remote Sensing Symposium, 2018.

@inproceedings{helber2018introducing,
  title={Introducing EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification},
  author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
  booktitle={IGARSS 2018-2018 IEEE International Geoscience and Remote Sensing Symposium},
  pages={204--207},
  year={2018},
  organization={IEEE}
}
