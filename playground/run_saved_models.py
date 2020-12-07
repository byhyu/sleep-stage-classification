#%%
from pysleep.data_generator import create_dataset,prepare_train_test
from tensorflow.keras.models import load_model
from pathlib import Path
from pysleep.models.base_model import  BaseModel
import numpy as np
#%%
model_path = Path(r'C:\Users\hyu\github-repos\LearnFromSleepData\saved_models\cnn3heads_origin.h5')
model = load_model(model_path)
#%%

train, val, test = prepare_train_test()
X, y = create_dataset(train[0:3])
val_X, val_y = create_dataset(val[0:2])
y_pred = []
#%%
y_pred = model.predict([val_X, val_X, val_X])
#%%
# for i, y in enumerate(val_y):
#     X = val_X[i,:]
#     print(X)
#     # Xs = np.array([X,X,X])
#     yp = model.predict([X,X,X])
#     y_pred.append(yp)
#%%

y_pred1 = np.argmax(y_pred, axis=1)
#%%
y_true = val_y.astype('int64')
#%%
import matplotlib.pyplot as plt

# def plot_roc_auc(y_preds, y_tests, model_name=""):
#     plt.figure()
#     y_preds_oh = convert_to_onehot(y_preds)
#     y_tests_oh = convert_to_onehot(y_tests)
#     print(y_preds_oh)
#     colors = ['', 'aqua', 'darkorange', 'cornflowerblue','', 'deeppink']
#     labels = ['', 'Sleep stage W', 'Sleep stage 1', 'Sleep stage 2', '', 'Sleep stage R']
#     for c in [1, 2, 3, 5]:
#         fpr, tpr, _ = roc_curve(y_tests_oh[:, c], y_preds_oh[:, c])
#         roc_auc = auc(fpr, tpr)
#         plt.plot(fpr, tpr, label=f"{model_name} (AUC of class {labels[c]} = {roc_auc:0.2f})", color = colors[c])
#     plt.legend(loc="lower right")
# #     plt.show()
#     plt.savefig("./plots/" + model_name + "_auc.png")

# from tensorflow.ke
m1 = BaseModel()

m1.plot_roc_auc(y_preds=y_pred1, y_tests=y_true, classes_to_plot=[0,1,2])
# m.plot_confusion_matrix()
#%%
def convert_to_onehot(array):
    n_values = np.max(array) + 1
    return np.eye(n_values)[array]


from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
classes_to_plot=[0,1,2,3,4]
y_tests_oh = convert_to_onehot(y_true)
y_preds_oh = convert_to_onehot(y_pred1)

colors = ['red', 'aqua', 'darkorange', 'cornflowerblue', 'green', 'deeppink']
labels = ['Sleep stage W', 'Sleep stage 1', 'Sleep stage 2', 'Sleep stage 3', 'Sleep stage R']
plt.figure()
for c in classes_to_plot:
    fpr, tpr, _ = roc_curve(y_tests_oh[:, c], y_pred[:, c])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr)#, label=f"cnn3heads_origin+(AUC of class {labels[c]} = {roc_auc:0.2f})")#color=colors[c]
# plt.legend(loc="lower right")
plt.show()

#%%
# plt.figure()
fig, ax = plt.subplots(1,1)
plt.plot([1,2,3], [1,4,5])
plt.show()

# y_pred1 = model.predict_classes([val_X,val_X,val_X])
