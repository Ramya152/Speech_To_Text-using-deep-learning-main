# Speech Classification with PyTorch and torchaudio
## Overview
This repository contains code for training and evaluating a speech classification model using PyTorch and torchaudio. The model is designed to classify spoken words from the Google Speech Commands dataset.

## Dataset
The Google Speech Commands dataset is a collection of short audio clips of spoken words, consisting of 30 different words such as "yes," "no," "up," "down," etc. The dataset is available for download from the TensorFlow website.

## Downloading the Dataset
To download and extract the dataset, follow these steps:

Run the command:

arduino
Copy code
!wget -O speech_commands_v0.01.tar.gz http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
Extract the downloaded tarball:

diff
Copy code
!tar xzf speech_commands_v0.01.tar.gz
## Code Structure
The code is organized into several Jupyter notebooks, each focusing on a specific aspect of the project:

01_load_dataset.ipynb: This notebook is used to load and explore the dataset. It provides an overview of the data structure and sample audio files.

02_preprocess_data.ipynb: This notebook preprocesses the audio data by extracting Mel Frequency Cepstral Coefficients (MFCCs), which are commonly used features in speech processing tasks.

03_create_dataset.ipynb: Here, custom dataset classes are created using PyTorch's torch.utils.data.Dataset interface. These classes facilitate the loading and preprocessing of data for training and evaluation.

04_build_model.ipynb: This notebook defines the architecture of the speech classification model. It includes the implementation of an RNN module (either LSTM or GRU), followed by a linear layer and a softmax activation function.

05_train_model.ipynb: The final notebook trains and evaluates the model on the training, validation, and test sets. It includes the training loop, evaluation metrics computation, and visualization of model performance.

## Usage
To use the code, follow these steps:

Install the required dependencies listed in the requirements.txt file:

## Copy code
pip install -r requirements.txt
Execute the Jupyter notebooks in the order mentioned above, following the instructions provided in each notebook.

## Model Architecture
The speech classification model consists of a recurrent neural network (RNN) module, which can be either a Long Short-Term Memory (LSTM) or a Gated Recurrent Unit (GRU). The input to the model is a sequence of MFCC features extracted from audio samples. The RNN module is followed by a linear layer and a softmax activation function to produce class probabilities.

## Evaluation
The trained model is evaluated on a separate test set using standard evaluation metrics such as accuracy and loss. Additionally, a confusion matrix is generated to visualize the model's performance across different classes.

