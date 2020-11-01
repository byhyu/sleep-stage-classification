# Set up development environment
## Option 1: Use Docker

- Build docker image: `docker build -t pyspark:latest -f Dockerfile .`
- To start the docker in interactive mode:
  Run `docker run -p 8888:8888 -it pyspark /bin/bash`
- To run notebook within docker: run `jupyter notebook`.

- More details can be found at: https://github.com/jupyter/docker-stacks

Info about Flint:
https://github.com/twosigma/flint/tree/master/python
Follow the instructions to install

Info about MNE:
https://mne.tools/stable/overview/cookbook.html

## Option 2: create virtual environments
Use `conda` or `anaconda`:
1. install `anaconda`
2. in `anaconda prompt`, navigate to `LearnFromSleep` project.
`conda env create -f environment.yaml`
3. activate environment:
`conda activate pysleep`

`PyCharm` IDE provides handy features to accelerate code development.

For the deep neural network models, it's recommended to run with GPU. 
To check is GPU is utilized:
`import tensorflow as tf
tf.config.list_physical_devices('GPU')`

If the output is `[]`, that means GPU is not used. If GPU exists and is utilized, output will look something like this:
`[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`
# Code structure
High level code organization:
```
.
├── data
├── Dockerfile
├── docs
├── environment.yaml
├── playground
├── __pycache__
├── pysleep
├── README.md
├── saved_models
├── setup.py
```
`data` folder: raw data and processed data.  
`docs`: documentaion and references  
`pysleep`: python package developed for this project  
`saved_models`: trained model, saved for reuse.  
`setup.py`: installation script for `pysleep` package.  
`Dockerfile`: dockerfile to created isolated dev environment if choose to use Option 1 Docker to set up dev environment.  
`environment.yaml`: configuration file for `conda` if choose to use Option 2 Conda to set up dev environment.   
`playground`: scripts for eda, run models, plot etc.  
`README.md`: quick introduction.  
```
A more detailed view of code structure (files may change):
.
├── data
│ ├── physionet_sleep
│ ├── sleep-edf-database-expanded-1.0.0
│ └── sleep-edf-database-expanded-1.0.0.zip
├── Dockerfile
├── docs
  ├── CSE6250_project_2020Fall.pdf
│ └── Team38_LearningFromSleepData.pdf
├── environment.yaml
├── models.py
├── playground
│ ├── baseline_model.py
│ ├── eda.py
│ ├── experimnents.py
│ ├── explore_models.py
│ ├── LoadData.ipynb
│ ├── prepare_train_test_dataset.py
│ └── try_data_loader.py
├── pysleep
│ ├── data.py
│ ├── dhedfreader.py
│ ├── __init__.py
│ ├── models.py
│ ├── prepare_physionet.py
│ └── __pycache__
├── README.md
├── reference
│ └── deepsleepnet-master
├── saved_models
│ ├── base_dnn_model.h5
│ └── dnn_model2.h5
├── setup.py
```
# Dataset
1. [PhysioNet](https://www.physionet.org/content/sleep-edfx/1.0.0/)
Online data viewer:
https://archive.physionet.org/cgi-bin/atm/ATM

Papers using this dataset:
- Huy Phan, Fernando Andreotti, Navin Cooray, Oliver Y. Chén, and Maarten De Vos. Joint Classification and Prediction CNN Framework for Automatic Sleep Stage Classification. IEEE Transactions on Biomedical Engineering, vol. 66, no. 5, pp. 1285-1296, 2019
https://github.com/pquochuy/MultitaskSleepNet

- https://github.com/SandraPrestel/deep-sleep-transfer


## Software for viewing EDF and EDF+ files
Both EDF and EDF+ formats are free and can be viewed using free software such as:

2. ISRUC_Sleep
https://sleeptight.isr.uc.pt/ISRUC_Sleep/

Polyman (for MS-Windows only; for details, please follow the link)
EDFbrowser (for Linux, Mac OS X, and MS-Windows; at www.teuniz.net)
LightWAVE and the PhysioBank ATM, platform-independent web applications from PhysioNet
WAVE and other applications for Linux, Mac OS X, and MS-Windows in the WFDB Software Package, also from PhysioNet

## Prepare Data
Transform EEG data into numpy array or pandas DataFrame:
https://github.com/Zhao-Kuangshi/sleep-edf-converter/blob/master/annotation_convertor.py

Note: I have trouble processing this file:
`sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4362F0-PSG.edf`
# References
## Collection of Paper with code
https://paperswithcode.com/task/sleep-stage-detection
https://github.com/SuperBruceJia/EEG-DL

## DeepSleepNet
https://github.com/akaraspt/deepsleepnet

## Blogs and articles
This Medium article presents a simple CNN model that is easy to reproduce.
https://towardsdatascience.com/sleep-stage-classification-from-single-channel-eeg-using-convolutional-neural-networks-5c710d92d38e

# Task breakup
## Timeline

## Tasks
- Setup environment
    - 
- Find dataset
- Data ETL (raw dataset to numpy arrays)
    - raw dataset to numpy arrays
    - data + lables
    - train, val, test split
- Feature engineering
    - automatic FE? (i.e. use CNN to extract features)
    - handcraft features
    - normalization?
    - data augmentation? (i.e. add noises)
- Data imbalance
    - SMOTE
    - oversampling, 
- Explore model archicture
    - conventional ML models, SVM, tree based models
    - Deep networks, CNN, RNN, LSTM
- Visualize
- Write report or paper if time permits

# Train models
## track model performance using `Tensorboard`
In command line:
`tensorboard --logdir playground\logs\fit`

