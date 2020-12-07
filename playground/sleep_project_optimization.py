import os
from os import walk

# from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools

import mne
# from mne.datasets.sleep_physionet.age import fetch_data
from mne.time_frequency import psd_welch

from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
# from sklearn.metrics import RocCurveDisplay

# from sklearn.metrics import plot_roc_curve
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, LabelBinarizer

# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def read_raw_and_annotation_data(raw_data_name, annotation_name, mapping, should_plot=False):
    '''Returns a raw object and the annotation object.
    Input:
        - raw_data_name: string of raw_data file. Name endedd with PSG.edf.
        - annotation_name: string of annotation file. Name ended with Hypnogram.edf.
        - should_plot: plot the data if set to true for debug purpose.
    '''
    raw_train = mne.io.read_raw_edf(raw_data_name)
    annot_train = mne.read_annotations(annotation_name)

    raw_train.set_annotations(annot_train, emit_warning=False)
    raw_train.set_channel_types(mapping)

    # plot some data
    if should_plot:
        raw_train.plot(duration=60, scalings='auto')
    return raw_train, annot_train


def get_rawdata_annotation_filenames(dir):
    '''Returns a dict of all key: (raw_data, annotation) map assuming the first 7 characters are the same.
    '''
    file_pairs = {}
    for (dirpath, dirnames, filenames) in walk(dir):
        break       
    for file in filenames:
        key = file[:7]
        if key not in file_pairs:
            file_pairs[key] = ['', '']
        if file.endswith('Hypnogram.edf'):
            file_pairs[key][1] = file
        elif file.endswith('PSG.edf'):
            file_pairs[key][0] = file
    return file_pairs


def extract_events_plot(raw_train, annot_train, chunk_duration, event_id):
    events_train, _ = mne.events_from_annotations(
    raw_train, event_id=event_id, chunk_duration=chunk_duration)

    fig = mne.viz.plot_events(events_train, event_id=event_id,
                          sfreq=raw_train.info['sfreq'],
                          first_samp=events_train[0, 0])
    stage_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def extract_epochs(raw_data_name, annotation_name, chunk_duration, mapping,event_id):
    raw = mne.io.read_raw_edf(raw_data_name)
    annot = mne.read_annotations(annotation_name)
    annot.crop(annot[1]['onset'] - 300 * 60,
               annot[-2]['onset'] + 300 * 60)
    raw.set_annotations(annot, emit_warning=False)
    raw.set_channel_types(mapping)
    
    events, _ = mne.events_from_annotations(
    raw, event_id=event_id, chunk_duration=chunk_duration)
    tmax = 30. - 1. / raw.info['sfreq']  # tmax in included
    epochs = mne.Epochs(raw=raw, events=events,
                        event_id=event_id, tmin=0., tmax=tmax, baseline=None)
    return epochs


def eeg_power_band(epochs):
    """EEG relative power band feature extraction.

    This function takes an ``mne.Epochs`` object and creates EEG features based
    on relative power in specific frequency bands that are compatible with
    scikit-learn.

    Parameters
    ----------
    epochs : Epochs
        The data.

    Returns
    -------
    X : numpy array of shape [n_samples, 5]
        Transformed data.
    """
    # specific frequency bands
    FREQ_BANDS = {"delta": [0.5, 4.5],
                  "theta": [4.5, 8.5],
                  "alpha": [8.5, 11.5],
                  "sigma": [11.5, 15.5],
                  "beta": [15.5, 30]}

    psds, freqs = psd_welch(epochs, picks='eeg', fmin=0.5, fmax=30.)
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    return np.concatenate(X, axis=1)


def data_split(file_map, file_pairs, seed=0, training_ratio=0.9, cv_fold=5):
    np.random.seed(seed)
    file_count = len(file_map)
    fold_size = int(len(file_map) * training_ratio / cv_fold)
    training_sets = [[] for _ in range(cv_fold)]
    test_sets = []
    for key, (v1, v2) in file_pairs.items():
        if not v1 or not v2:
            continue
        if np.random.random_sample() <= training_ratio:
            for training_fold in training_sets:
                if len(training_fold) < fold_size:
                    training_fold.append(key)
        else:
            test_sets.append(key)
    return training_sets, test_sets


def grid_search(cv_fold,training_sets,DATA_PATH,file_pairs,RANDOM_STATE):
    ### Grid search for hyperparameter tuning
    cv_accs_ave_mat = np.zeros([7,7])
    m=0
    n=0
    for estimators in range(50,151,15):
        for depth in range(2,22,3):
    
            # RandomForestClassifier
            pipe = make_pipeline(FunctionTransformer(eeg_power_band, validate=False),
                                  RandomForestClassifier(n_estimators=estimators, random_state=RANDOM_STATE, max_depth=depth))
            
            # pipe = make_pipeline(FunctionTransformer(eeg_power_band, validate=False),
            #                       RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8))
            
            # pipe = make_pipeline(FunctionTransformer(eeg_power_band, validate=False),
            #                      SVC(gamma=1e-5, C=0.5))
            
            # pipe = make_pipeline(FunctionTransformer(eeg_power_band, validate=False),
            #                       LogisticRegression(C=20,random_state=RANDOM_STATE))
            
            
            left_out = cv_fold - 1
            # Train with cross validation
            for _ in range(cv_fold):
                # Train
                for i in range(cv_fold):
                    if i == left_out:
                        continue
                    count = 0
                    for key in training_sets[i]:
                        epochs_train = extract_epochs(os.path.join(DATA_PATH, file_pairs[key][0]), os.path.join(DATA_PATH, file_pairs[key][1]), 30)
            
                        if count == 0:
                            epochs_train1 = epochs_train
                        else:
                            epochs_train1 = mne.concatenate_epochs((epochs_train1,epochs_train))
                        count += 1
                        epochs_train1.drop_bad()
                        print('-----')
                        print(len(epochs_train1))
            
                    pipe.fit(epochs_train1, epochs_train1.events[:, 2])
                    
                cv_accs = []
                # cv_roc_aucs = []
                for key in training_sets[left_out]:
                    epochs_test = extract_epochs(os.path.join(DATA_PATH, file_pairs[key][0]), os.path.join(DATA_PATH, file_pairs[key][1]), 30)
                    y_pred = pipe.predict(epochs_test)
                    y_test = epochs_test.events[:, 2]
                    acc = accuracy_score(y_test, y_pred)
                    cv_accs.append(acc)
                    
            #         roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
            #         cv_roc_aucs.append(roc_auc)
                cv_accs_ave = np.average(cv_accs)
                cv_accs_ave_mat[m][n] = cv_accs_ave
                print('=======')
                print("CV accuracy score: {}".format(cv_accs_ave))
            #     print("CV ROC AUC score: {}".format(np.average(cv_roc_aucs)))
                left_out -= 1
            
            n += 1
        m += 1
        if n>6:
            n = 0


def convert_to_onehot(array):
    n_values = np.max(array) + 1
    return np.eye(n_values)[array]


def plot_roc_auc(y_preds, y_tests, model_name=""):
    plt.figure()
    y_preds_oh = convert_to_onehot(y_preds)
    y_tests_oh = convert_to_onehot(y_tests)
    # y_preds_oh = y_preds
    # y_tests_oh = y_tests
    print(y_preds_oh[0])
    colors = ['', 'aqua', 'darkorange', 'cornflowerblue','', 'deeppink']
    labels = ['', 'Sleep stage W', 'Sleep stage 1', 'Sleep stage 2', '', 'Sleep stage R']
    for c in [1, 2, 3, 5]:
        fpr, tpr, _ = roc_curve(y_tests_oh[:, c], y_preds_oh[:, c])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"(AUC of class {labels[c]} = {roc_auc:0.2f})", color = colors[c])
    plt.legend(loc="lower right")
#     plt.show()
    plt.savefig("./plots/" + model_name + "_auc.png")
    

def plot_confusion_matrix(cm, class_names, figurename):

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(i, j, "{:.2f}".format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(figurename)
    

def multiclass_roc_auc_score(y_test, y_pred, average=None):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

def main():
    
    DATA_PATH = '../project/sleep-edf-database-expanded-1.0.0/sleep-cassette/'
    mapping = {'EOG horizontal': 'eog',
           'Resp oro-nasal': 'misc',
           'EMG submental': 'misc',
           'Temp rectal': 'misc',
           'Event marker': 'misc'}

    file_pairs = get_rawdata_annotation_filenames(DATA_PATH)

    event_id = {'Sleep stage W': 1,
                'Sleep stage 1': 2,
                'Sleep stage 2': 3,
                'Sleep stage 3': 5}

    temp = []
    file_pairs1 = {}
    for row in file_pairs:
        temp.append(row)
    for row in temp[0:20]:
        file_pairs1[row]=file_pairs[row]
    
    file_pairs = file_pairs1
    
    RANDOM_STATE = 42
    cv_fold = 1
    training_sets, test_sets = data_split(file_pairs, file_pairs, seed=0, training_ratio=0.8, cv_fold=cv_fold)

    ### Training
    
    estimators = 50
    depth = 11
    # scores = []
    # for depth in range(2,22,3):
        # RandomForestClassifier
    pipe = make_pipeline(FunctionTransformer(eeg_power_band, validate=False),
                          RandomForestClassifier(n_estimators=estimators, random_state=RANDOM_STATE, max_depth=depth))
    
    count = 0
    for key in training_sets[0]:
        epochs_train = extract_epochs(os.path.join(DATA_PATH, file_pairs[key][0]), os.path.join(DATA_PATH, file_pairs[key][1]), 30, mapping, event_id)
    
        if count == 0:
            epochs_train1 = epochs_train
        else:
            epochs_train1 = mne.concatenate_epochs((epochs_train1,epochs_train))
        count += 1
        epochs_train1.drop_bad()
        print('-----')
        print(len(epochs_train1))
    
    pipe.fit(epochs_train1, epochs_train1.events[:, 2])    

    filename = 'saved_models/random_forest1.sav'
    pickle.dump(pipe, open(filename, 'wb'))
    
    # loaded_model = pickle.load(open(filename, 'rb'))

    ### Test

    accs = []
    cm_total = np.zeros((4, 4))
    y_preds = []
    y_preds_proba = []
    y_tests = []
    for i in range(len(test_sets)):
        key = test_sets[i]
        epochs_test = extract_epochs(os.path.join(DATA_PATH, file_pairs[key][0]), os.path.join(DATA_PATH, file_pairs[key][1]), 30, mapping, event_id)
        y_pred = pipe.predict(epochs_test)
        y_pred_proba = pipe.predict_proba(epochs_test)
        y_test = epochs_test.events[:, 2]
        acc = accuracy_score(y_test, y_pred)
        accs.append(acc)
        cm = confusion_matrix(y_test, y_pred)
        cm_total += cm
        y_preds.extend(y_pred.tolist())
        y_preds_proba.extend(y_pred_proba.tolist())    
        y_tests.extend(y_test.tolist())

    plot_confusion_matrix(cm_total, ['Sleep stage W', 'Sleep stage 1', 'Sleep stage 2', 'Sleep stage R'], 'plots/rf1_cm.png')

    # y_preds_oh = convert_to_onehot(y_preds)
    # y_tests_oh = convert_to_onehot(y_tests)
    # print(y_preds)
    # print(y_preds_oh[50:100, :])
    # plot_roc_auc(y_preds, y_tests, "random_forest1")

    print(accs)
    print("Test accuracy score: {}".format(np.average(accs)))
    print("Confusion matrix\n")
    print(cm_total)
    # scores.append(classification_report(y_tests, y_preds, target_names=event_id.keys()))
    # print(scores)
    print(classification_report(y_tests, y_preds, target_names=event_id.keys()))
    
    print('ROC AUC ------')
    print(multiclass_roc_auc_score(y_preds, y_tests,average = None))
    print('micro',multiclass_roc_auc_score(y_preds, y_tests,average = 'micro'))
    print('macro',multiclass_roc_auc_score(y_preds, y_tests,average = 'macro'))
    print('samples',multiclass_roc_auc_score(y_preds, y_tests,average = 'samples'))
    print('weighted',multiclass_roc_auc_score(y_preds, y_tests,average = 'weighted'))
    
if __name__ == "__main__":
    main()