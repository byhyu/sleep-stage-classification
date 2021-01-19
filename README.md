# Learning From Sleep Data
## Introduction
Sleep is vital to human health. The quality of sleep can be used as an indicator or precursor of certain diseases, such as Alzheimer [1] and Parkinson’s disease [2]. According to American Academy of Sleep Medicine (AASM) manual [3], sleep can be divided into five stages: Wake (W), Non-Rapid Eye Movement stages N1 (for drowsiness or transitional sleep), N2 (for light sleep), and N3 (for deep sleep), and Rapid Eye Movement (REM), which can be determined by analyzing polysomnograms (PSG) data of patients during sleep. PSG data is a collection of data collected from various sensors, including electroencephalography (EEG), Electromyogram (EMG) and Electrocardiogram (ECG) recordings. 
Identification of sleep stages enables diagnosis of many sleep disorders. Currently, sleep stage classification relies heavily on manual inspection by well-trained physicians or technicians, which is expensive and tedious [4]. The growing need for accurate and fast sleep stage classification and limitations of manual labeling have led to an emerging research in automatic sleep stage classification (ASSC) systems. This project aimed to develop an end-to-end automatic ASSC system that takes a sequence of single channel EEG recordings and outputs a sequence of sleep stage labels. To this end, a big data analytics tool PySpark was used to extract, transform, and load (ETL) Sleep-EDF Database Expanded available at PhysioNet [18]. Several machine learning models, including random forest, decision tree, and K-nearest neighbours (KNN), were implemented with scikit-learn or mllib. Deep learning methods such as Multiple Layer Perceptron (MLP), Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) were also employed and run on Amazon Web Services (AWS). By performing a feature importance study and optimizing the hyperparameters through cross-validation, we were able to achieve an overall model accuracy of 93% (F1 score) and 95% (AUC), respectively, which is close to the best performance presented in the open literature and surpasses that of many past studies [4].
## Approach
The general workflow or pipeline of the proposed approach mainly comprises four essential steps, including feature engineering, training, evaluation, and deployment. A brief description of each step is given below.

### Feature engineering
Extraction of highly relevant features from the raw input data is the very first critical step in the whole workflow. EEG signals are time series data of local electrical potentials, which are gathered from electrodes placed on multiple regions of a subject’s scalp, each representing a data channel. The Rechtschaffen and Kales standard (R&K) rules and the American Academy of Sleep Medicine (AASM) define the criteria for stage scoring or staging of sleep for adults. Both R&K and AASM recommend the use of 30-second epochs of PSG signals for sleep staging. An expert scorer assigns one of the five stage names to each 30 seconds of the EEG data using the standard scoring rules.
There are three different feature representations pertaining to EEG signals, including raw EEG data, spectrogram, and expert-defined features. The raw EEG data can be considered as a three-dimensional tensor of n epochs, m channels in each epoch, and k (i.e., 30 multiplied with the sampling rate) data points in each epoch. The time series of EEG data can be converted into the frequency domain through Fourier transformation to obtain a spectrogram, which can be described with another three-dimensional tensor of n epochs, m (e.g., 29) sub-epochs of a 2-second duration with a 1-second overlap, and k (e.g., 257) frequency bins. Features can also be manually defined by experts who examine the both time series and spectrogram of the EEG data and consult the AASM rule sets. The expert-defined features will be used as the ground truth to assess the predictive accuracy of the machine learning models.

### Training
The training step is to select and train an appropriate classification model which can autonomously annotate the EEG data epochs using the standard stage names and the constructed features. We started with several machine learning algorithms, such as random forests and K-nearest neighbors (KNN), and then applied deep learning methods, including Concurrent Neural Networks (CNN), Recurrent Neural Networks (RNN), and a combination of both (CNN-RNN). We used a number of filters to convolve the feature matrix to produce preactivation feature maps and then invoked the Rectifier Linear Unit (ReLu) as non-linear activation functions before passing the features through a max-pooling layer to reduce the spatial size of the representation. We implemented the RNN formulation using the Long Short Term Memory (LSTM) method in Tensorflow and incorporated dropout regularization to avoid overfitting. For the hybrid CNN-RNN model, we used CNN to extract the spatial features from EEG, which is time invariant and independent in each step, and passed them to a RNN model, which learns the temporal dependency present of the spatial feature already extracted by CNN.  

### Evaluation
In the evaluation step, the data was split into training, validation, and test sets. We used 80% of the considered EEG data as the training data and the remaining 20% as the test data. For each classification method evaluated, we used a random search and then a grid search, along with cross-validation, to tune hyperparameters. The metrics to be used include precision, recall, ROC-AUC, F1 score, and confusion matrices. 

### Deployment
The deployment of the selected model was implemented on local machines and AWS. Sci-kit learn and mllib were used for the machine learning models (e.g., random forests and KNN), while TensorFlow was employed for the deep learning methods (e.g., CNN and LSTM).

## Set up development environment
### Option 1: Use Docker
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

### Option 2: create virtual environments
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
### Code structure
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
│ ├── RandomForest.ipynb
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
## How to run
Then entry point is the `playground` folder. `RandomForest.ipynb` contains initial exploration using random forest and nearest neighbors.
`sleep_project_optimized.py` contains model finetuning and furture feature engineeering.
`deeplearnibg` folder contains explorations with CNN and LSTM models. 

## Dataset
1. [PhysioNet](https://www.physionet.org/content/sleep-edfx/1.0.0/)
Online data viewer:
https://archive.physionet.org/cgi-bin/atm/ATM

Papers using this dataset:
- Huy Phan, Fernando Andreotti, Navin Cooray, Oliver Y. Chén, and Maarten De Vos. Joint Classification and Prediction CNN Framework for Automatic Sleep Stage Classification. IEEE Transactions on Biomedical Engineering, vol. 66, no. 5, pp. 1285-1296, 2019
https://github.com/pquochuy/MultitaskSleepNet

- https://github.com/SandraPrestel/deep-sleep-transfer


### Software for viewing EDF and EDF+ files
Both EDF and EDF+ formats are free and can be viewed using free software such as:

2. ISRUC_Sleep
https://sleeptight.isr.uc.pt/ISRUC_Sleep/

Polyman (for MS-Windows only; for details, please follow the link)
EDFbrowser (for Linux, Mac OS X, and MS-Windows; at www.teuniz.net)
LightWAVE and the PhysioBank ATM, platform-independent web applications from PhysioNet
WAVE and other applications for Linux, Mac OS X, and MS-Windows in the WFDB Software Package, also from PhysioNet

### Prepare Data
Transform EEG data into numpy array or pandas DataFrame:
https://github.com/Zhao-Kuangshi/sleep-edf-converter/blob/master/annotation_convertor.py




## Blogs and articles
This Medium article presents a simple CNN model that is easy to reproduce.
https://towardsdatascience.com/sleep-stage-classification-from-single-channel-eeg-using-convolutional-neural-networks-5c710d92d38e
https://github.com/CVxTz/EEG_classification

## Model Architecture
For single sleep epoch (30s by 100Hz = 3000 data points), need to encode into a vector/tensor.
For sleep epoch sequences (for example, 8H sleep represented by multiple 30s epochs), output a sequence of categories.

CNN, SVM, or other classification models for single sleep epoch encoding.
LSTM for sequence classification.

## Train models
### track model performance using `Tensorboard`
In command line:
`tensorboard --logdir playground\logs\fit`


## Task breakup
### Tasks
- Setup environment
    - 
- Find dataset
- Data ETL (raw dataset to numpy arrays)
    - raw dataset to numpy arrays
    - data + lables
    - train, val, test split
- Feature engineering
    - automati
c FE? (i.e. use CNN to extract features)
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
 
## References
1   Lim MM, Gerstner JR, Holtzman DM. The sleep-wake cycle and Alzheimer’s disease: what do we know? Neurodegener Dis Manag 2014;4:351–62. doi:10.2217/nmt.14.33
2 	Cooray N, Andreotti F, Lo C, et al. Detection of REM sleep behaviour disorder by automated polysomnography analysis. Clin Neurophysiol 2019;130:505–14. doi:10.1016/j.clinph.2019.01.011
3 	Berry RB, Brooks R, Gamaldo CE, et al. The AASM manual for the scoring of sleep and associated events. … Academy of Sleep … 2012.
4 	Aboalayon K, Faezipour M, Almuhammadi W, et al. Sleep stage classification using EEG signal analysis: A comprehensive survey and new investigation. Entropy 2016;18:272. doi:10.3390/e18090272
5 	Hassan AR, Bashar SK, Bhuiyan MIH. On the classification of sleep states by means of statistical and spectral features from single channel Electroencephalogram. In: 2015 International Conference on Advances in Computing, Communications and Informatics (ICACCI). IEEE 2015. 2238–43. doi:10.1109/ICACCI.2015.7275950
6 	Hassan AR, Hassan Bhuiyan MI. Automatic sleep scoring using statistical features in the EMD domain and ensemble methods. Biocybernetics and Biomedical Engineering 2016;36:248–55. doi:10.1016/j.bbe.2015.11.001

