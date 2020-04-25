
import pandas as pd
import functions as fu
import pathlib
import numpy as np
import csv
import os
import time
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.under_sampling import NeighbourhoodCleaningRule  # , RandomUnderSampler, OneSidedSelection, TomekLinks

os.chdir(str(pathlib.Path(__file__).parent.absolute()))

workspace_path = str(pathlib.Path(pathlib.Path(__file__).parent.absolute()).parent)
print(workspace_path)

area_used = str(input('Give the number of the area to process (2,3,7,8): '))

year = '/'
area = '/area_' + area_used

data_folder_path = workspace_path + area + '/Data' + year
labels_folder_path = workspace_path + area + '/Ground_Truth_Data' + year
plots_folder_path = workspace_path + area + '/Plots' + year
shapefiles_folder_path = data_folder_path + 'Shapefiles/'
separate_bands_folder_path = 'Separate_Bands/'
csv_path = data_folder_path + 'CSVs/'

metrics_list = ["Precision", "Recall", "F1 Score"]
indeces = ["ndvi"]


with open(data_folder_path + 'crops_names_and_id.csv', newline='') as f:
    reader = csv.reader(f)
    data = dict(reader)

crop_names_list = list(data.keys())
labels_nums = [int(x) for x in list(data.values())]
labels_nums.sort()
print(labels_nums)
print(crop_names_list)

raster_name_list = os.listdir(data_folder_path)
# print(raster_name_list)
raster_name_list = sorted([file for file in raster_name_list if file.endswith('.tif')])

new_list = []
for file_name in raster_name_list:
    new_list.append(file_name.split('_')[0])

new_list = list(set(new_list))
print("Raster files detected: ")
for i in new_list:
    print(i)
filling_mode = input("Enter filling mode for processing: ")

clean_dataset = input('Clean Dataset with Neighbourhood Cleaning Rule? (yes/no): ')
if clean_dataset == 'yes':
    cleaned = "_cleaned"
else:
    cleaned=""


txt_name = "obia_" + area_used + "_results_" + filling_mode + cleaned
file = open(txt_name, "w")
file.write(area[1:] + "\n\n")

no_bands = len(os.listdir(data_folder_path + 'Separate_Bands/'))

index = 'ndvi'
print("open csvs calculate metric ")
# #READ CSVs WITH STATISTICS, TAKE THE METRIC TO UTILIZE, CREATE DATA AND LABELS ARRAYS PER CROP
index_means = []
index_labels = []
for j, crop in enumerate(crop_names_list):
    date_array = []
    for i in range(no_bands):
        stats = pd.read_csv(data_folder_path + 'CSVs/' + crop + '_date_' + str(i + 1) + str('_' + index) + '_stats.csv')
        date_array.append(stats.values)
    #    crop_arrays.append(np.asarray(date_array))
    temp = np.squeeze(np.asarray(date_array))  # mean
    print(temp.shape)
    index_labels.append(np.ones(temp.shape[1]) * labels_nums[j])
    index_means.append(np.transpose(temp))
    print(index_means[j].shape)

# dict_of_indeces_with_data_and_labels_in_list[index] = [index_means, index_labels]
data = np.copy(index_means[0])
labels = np.copy(index_labels[0])
for i in range(1, len(index_labels)):
    data = np.concatenate((data, index_means[i]), axis=0)
    labels = np.concatenate((labels, index_labels[i]), axis=0)


print(data.shape)
print(labels.shape)
# %%
# CREATE TRAIN TEST DATASETS FOR ML CLASSIFIERS
train_test_ratio = 0.2


if clean_dataset == 'yes':
    file.write("UnderSampling: " + str(clean_dataset) + "\n\n")
    total_elements = len(labels)
    background_elements = len(labels[labels != 0])
    resample_dict = {0: int(background_elements)}

    rus = NeighbourhoodCleaningRule(sampling_strategy='all')
    # rus = TomekLinks(sampling_strategy='all')
    # rus = RandomUnderSampler(sampling_strategy="not minority")
    # rus = OneSidedSelection(sampling_strategy='all',n_seeds_S=1000)

    start_train = time.time()
    X_rus, y_rus = rus.fit_sample(data, labels)
    end_train = time.time()
    print("Balancing time: ", end_train - start_train)

    file.write("Balancing time: " + str(end_train - start_train) + "\n\n")

    for i in labels_nums:
        print("Before Class " + str(i) + " number of samples: ", len(labels[labels == i]))
        file.write("Before Class " + str(i) + " number of samples: " + str(
            len(labels[labels == i])) + "\n")
    file.write("\n")
    for i in labels_nums:
        print("After Class " + str(i) + " number of samples: ", len(y_rus[y_rus == i]))
        file.write("After Class " + str(i) + " number of samples: " + str(len(y_rus[y_rus == i])) + "\n")
    file.write("\n")

    X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(X_rus, y_rus, test_size=train_test_ratio,
                                                                    random_state=42, stratify=y_rus)

else:
    file.write("No UnderSampling \n\n")
    print("No UnderSampling Train-Test splitting")
    X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(data, labels, test_size=train_test_ratio,
                                                                    random_state=42, stratify=labels)

print("mlp")
file.write("\n\nMLP \n\n")
# %%   28,14,7  200,100,50
mlp = MLPClassifier(hidden_layer_sizes=(28, 14), max_iter=1000, activation='relu', random_state=42, verbose=True,
                    batch_size=10)
start_train = time.time()
mlp.fit(X_train_ml, y_train_ml)
end_train = time.time()
print("MLP train time: ", end_train - start_train)
file.write("MLP train time: " + str(end_train - start_train) + "\n")

start_test = time.time()
y_pred_mlp = mlp.predict(X_test_ml)
end_test = time.time()
print("MLP test time: ", end_test - start_test)

file.write("MLP test time: " + str(end_test - start_test) + "\n")
file.write("MLP Final loss: " + str(mlp.loss_) + "\n")
print(mlp.loss_)

start_test = time.time()
y_train_pred_mlp = mlp.predict(X_train_ml)
end_test = time.time()
print("Train_accuracy test time: ", end_test - start_test)

cm = confusion_matrix(y_pred_mlp, y_test_ml)
train_cm = confusion_matrix(y_train_pred_mlp, y_train_ml)
print("Test Accuracy of MLPClassifier : ", fu.accuracy(cm))
print("Train Accuracy of MLPClassifier : ", fu.accuracy(train_cm))
file.write("MLP Classifier Test Accuracy: " + str(fu.accuracy(cm)) + "\n")
file.write("MLP Classifier Train Accuracy: " + str(fu.accuracy(train_cm)) + "\n")

stat_res = precision_recall_fscore_support(y_test_ml, y_pred_mlp, labels=labels_nums)
print(stat_res)
# fu.print_confusion_matrix(cm,labels_nums)

file.write("MLP Precision, Recall, F1 score \n\n")
for i, met in enumerate(metrics_list):
    file.write(met + "\n")
    file.write(str(stat_res[i]) + "\n")
# In[16]:

print("RF")
file.write("\n\nRF \n\n")

rf_obj = RandomForestClassifier(random_state=0)
start_train = time.time()
rf_obj.fit(X_train_ml, y_train_ml)
end_train = time.time()
print("RF train time: ", end_train - start_train)
file.write("RF train time: " + str(end_train - start_train) + "\n")

start_test = time.time()
y_predict_rf = rf_obj.predict(X_test_ml)
end_test = time.time()
print("RF test time: ", end_test - start_test)
file.write("RF test time: " + str(end_test - start_test) + "\n")

rf_y_pred_train_ml = rf_obj.predict(X_train_ml)

cm_rf = confusion_matrix(y_predict_rf, y_test_ml)
cm_train_rf = confusion_matrix(rf_y_pred_train_ml, y_train_ml)

print("Test Accuracy of RF Classifier : ", fu.accuracy(cm_rf))
print("Train Accuracy of RF Classifier : ", fu.accuracy(cm_train_rf))
file.write("RF Classifier Test Accuracy: " + str(fu.accuracy(cm_rf)) + "\n")
file.write("RF Classifier Train Accuracy: " + str(fu.accuracy(cm_train_rf)) + "\n")

stat_res_rf = precision_recall_fscore_support(y_test_ml, y_predict_rf, labels=labels_nums)
print(stat_res_rf)
# fu.print_confusion_matrix(cm_rf,labels_nums)
file.write("\nRF Precision, Recall, F1 score \n\n")
for i, met in enumerate(metrics_list):
    file.write(met + "\n")
    file.write(str(stat_res_rf[i]) + "\n")

rf_obj_importance = rf_obj.feature_importances_
file.write("RF importancese: \n")
file.write(str(rf_obj_importance) + "\n")

print(rf_obj_importance)
# In[17]:

print("svm")
file.write("\n\nSVM \n\n")

svm_clf = SVC(gamma='scale', C=25)
start_train = time.time()
svm_clf.fit(X_train_ml, y_train_ml)
end_train = time.time()
print("SVM train time: ", end_train - start_train)
file.write("SVM train time: " + str(end_train - start_train) + "\n")

start_test = time.time()
y_predict_svm = svm_clf.predict(X_test_ml)
end_test = time.time()
print("SVM test time: ", end_test - start_test)
file.write("SVM test time: " + str(end_test - start_test) + "\n")

svm_y_pred_train_ml = svm_clf.predict(X_train_ml)

cm_svm = confusion_matrix(y_predict_svm, y_test_ml)
cm_train_svm = confusion_matrix(svm_y_pred_train_ml, y_train_ml)

print("Test Accuracy of SVM Classifier : ", fu.accuracy(cm_svm))
print("Train Accuracy of SVM Classifier : ", fu.accuracy(cm_train_svm))
file.write("SVM Classifier Test Accuracy: " + str(fu.accuracy(cm_svm)) + "\n")
file.write("SVM Classifier Train Accuracy: " + str(fu.accuracy(cm_train_svm)) + "\n")

stat_res_svm = precision_recall_fscore_support(y_test_ml, y_predict_svm, labels=labels_nums)
print(stat_res_svm)
# fu.print_confusion_matrix(cm_svm,labels_nums)
file.write("SVM Precision, Recall, F1 score \n\n")
for i, met in enumerate(metrics_list):
    file.write(met + "\n")
    file.write(str(stat_res_svm[i]) + "\n")

print("KNN")
file.write("\n\nKNN - 3 \n\n")
# In[18]:


neigh_num = 3
knn_clf = KNeighborsClassifier(n_neighbors=neigh_num)

start_train = time.time()
knn_clf.fit(X_train_ml, y_train_ml)
end_train = time.time()

print("KNN train time: ", end_train - start_train)
file.write("KNN train time: " + str(end_train - start_train) + "\n")

start_test = time.time()
y_predict_knn = knn_clf.predict(X_test_ml)
end_test = time.time()

print("KNN test time: ", end_test - start_test)
file.write("KNN test time: " + str(end_test - start_test) + "\n")

knn_y_pred_train_ml = knn_clf.predict(X_train_ml)

cm_knn = confusion_matrix(y_predict_knn, y_test_ml)
cm_train_knn = confusion_matrix(knn_y_pred_train_ml, y_train_ml)

print("Test Accuracy of KNN Classifier : ", fu.accuracy(cm_knn))
print("Train Accuracy of KNN Classifier : ", fu.accuracy(cm_train_knn))
file.write("KNN Classifier Test Accuracy: " + str(fu.accuracy(cm_knn)) + "\n")
file.write("KNN Classifier Train Accuracy: " + str(fu.accuracy(cm_train_knn)) + "\n")
stat_res_knn = precision_recall_fscore_support(y_test_ml, y_predict_knn, labels=labels_nums)
print(stat_res_knn)
# fu.print_confusion_matrix(cm_knn,labels_nums)
file.write("KNN Precision, Recall, F1 score \n\n")
for i, met in enumerate(metrics_list):
    file.write(met + "\n")
    file.write(str(stat_res_knn[i]) + "\n")
# In[19]:


# TRAIN-TEST DATASETS FOR RMSE METHOD
# GETTING THE MEANS OF EACH CLASS AND USING RMSE WE CLASSIFY THE TEST DATA TO THE MINIMUM ERROR CLASS

print("RMSE")
file.write("\n\nRMSE \n\n")

means = []
for label in labels_nums:
    means.append(np.mean(X_train_ml[y_train_ml == label], axis=0))

for line in range(len(means)):
    plt.plot(means[line])
plt.legend(crop_names_list)
plt.savefig(plots_folder_path + 'Lines_' + filling_mode + cleaned + '.png')

results_rmse = []
results_mae = []
for vec in X_test_ml:
    rmse = []
    mae = []
    for crop_graph in means:
        mae.append((mean_absolute_error(crop_graph, vec)))
        rmse.append((mean_squared_error(crop_graph, vec)))
    results_rmse.append(labels_nums[rmse.index(min(rmse))])
    results_mae.append(labels_nums[mae.index(min(mae))])


cm_rmse = confusion_matrix(y_test_ml, results_rmse)
stat_res_rmse = precision_recall_fscore_support(y_test_ml, results_rmse, labels=labels_nums)
print(stat_res_rmse)
# fu.print_confusion_matrix(cm_rmse,labels_nums)
print("Test Accuracy of RMSE Classifier : ", fu.accuracy(cm_rmse))
file.write("RMSE Classifier Test Accuracy: " + str(fu.accuracy(cm_rmse)) + "\n")
file.write("RMSE Precision, Recall, F1 score \n\n")
for i, met in enumerate(metrics_list):
    file.write(met + "\n")
    file.write(str(stat_res_rmse[i]) + "\n")

cm_mae = confusion_matrix(y_test_ml, results_mae)
stat_res_mae = precision_recall_fscore_support(y_test_ml, results_mae, labels=labels_nums)
print(stat_res_mae)
# fu.print_confusion_matrix(cm_mae,labels_nums)
print("Test Accuracy of MAE Classifier : ", fu.accuracy(cm_mae))
file.write("MAE Classifier Test Accuracy: " + str(fu.accuracy(cm_mae)) + "\n")
file.write("MAE Precision, Recall, F1 score \n\n")
for i, met in enumerate(metrics_list):
    file.write(met + "\n")
    file.write(str(stat_res_mae[i]) + "\n")

file.close()
