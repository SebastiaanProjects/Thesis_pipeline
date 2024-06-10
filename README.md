# Thesis_pipeline

```markdown
# My Project
For this project my aim is to combine CenterNet and Masked autoencoders pretraining in object recognition on a timeseries dataset.

For this project I have been given a dataset containing the accelerations and speed for crabplovers over 10 second time intervals. There is an unlabelled dataset, used for pretraining, and a labelled dataset used for supervised learning. The goal is to find the influence of the pretraining on the performance of CenterNet.

The CenterNet model has been created to work with Timeseries data containing four variables. The data is encoded CenterNet's backbone and then passed onto a heatmaphead, sizemaphead and offsethead as a featuremap. The heatmaphead aims to predict the middlepoint of the behaviour which is showcased by the crabplovers, for each behavioural class. The sizemaphead creates a sizemap indicating the duration of each behaviour for point p in time. The offset head reconstructs the difference between the original locations and the locations after downsampling via stride took place. Once these maps are combined using peakextraction and timeline reconstruction methods, the output can be compared to the original labelled data.

For pretraining Masked autoencoders is used. This method is, like CenterNet, originally meant for 2d imagery data, but is modified to match the needs for this project. For this implementation of Masked autoencoders the timeseries data is diveded into segments of which 80% will be randomly masked. Then the encoder will make a latent representation of the data and feed that to a architecturally simple decoder. This decoder will try to reconstruct the original dataset, allowing for the encoder to try to capture important patterns and connections within the dataset.

When combining the two models in pipeline_with_pretrain the Decoder is replaced by the CenterNet model. The encoder then functions as the backbone of the CenterNet model, and cooperates with the detectionhead. 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Installation

To set up the project locally, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/SebastiaanProjects/Thesis_pipeline.git
cd your-repository
pip install -r requirements.txt

Using masked auto encoders to pretrain a timeseries implementation of CenterNet

## usage

CenterNet_utils                 utility functions for CenterNet model
data_composer                   pipeline for composing and preprocessing data
Data_extraction_utils           utility functions for extractingand preprocessing data
MAE_utils                       utility functions for pretraining with Masked Autoencoders
original_pipeline_no_pretrain   script for running CenterNet without MAE-pretraining
pipeline_with_pretraining       script for running CenterNet in combination with MAE pretraining
pre_train.py                    script for running the Masked Autoencoders and creating the pretrained encoder
