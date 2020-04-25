#!/usr/bin/env python
# coding: utf-8

# In[26]:


import matplotlib.pyplot as plt
import numpy as np
from rasterio import plot
import pandas as pd
import os
from osgeo import gdal
import csv
import time

from math import sqrt

import rasterstats as rs
import pathlib
os.chdir(str(pathlib.Path(__file__).parent.absolute()))
import functions as fu


from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.under_sampling import RandomUnderSampler, OneSidedSelection, TomekLinks, NeighbourhoodCleaningRule
# import gc


workspace_path = str(pathlib.Path(pathlib.Path(__file__).parent.absolute()).parent)
print(workspace_path)
# In[10]:


# year = '_2019/'
# year = '_2018/'
# area = ''

area_used = str(input('Give the number of the area to process (2,3,7,8): '))

year = '/'
area = '/area_' + area_used

data_folder_path = workspace_path + area + '/Data' + year
labels_folder_path = workspace_path + area + '/Ground_Truth_Data' + year
plots_folder_path = workspace_path + area + '/Plots' + year



shapefiles_folder_path = data_folder_path + 'Shapefiles/'
separate_bands_folder_path = 'Separate_Bands/'
csv_path = data_folder_path + 'CSVs/'


# selected_indeces = input('Write Indeces(seperated with 1 space, options: ndvi, evi): ')
# indeces = selected_indeces.split()
# print(indeces)
indeces = ["ndvi"]
# In[3]:

txt_name = str(input('Name of the file to write results: '))
file = open(txt_name,"w")
file.write(area[1:] + "\n\n")

mode_to_process = str(input('Mode to process: (spat/temp/mixed/none): '))


shapefiles_name_list = os.listdir(data_folder_path + 'Shapefiles/')
shapefiles_name_list = sorted([file for file in shapefiles_name_list if file.endswith('.shp')])

new_list = []
for file_name in shapefiles_name_list:
    new_list.append(file_name.split('_')[0])

new_list = list(set(new_list))
print("Threshold shapelifes detected: ")
for i in new_list:
    print(i)
threshold = input("Enter threshold of number of pixels per parcel for processing: ")
file.write("Threshold used for parcel detection: " +  + "\n\n")
shapefiles_name_list = sorted([file for file in shapefiles_name_list if file.startswith(threshold)])


with open(data_folder_path + 'crops_names_and_id.csv', newline='') as f:
    reader = csv.reader(f)
    data = dict(reader)

crop_names_list = list(data.keys())
labels_nums = [ int(x) for x in list(data.values()) ]
print(shapefiles_name_list)
print(labels_nums)
print(crop_names_list)


# In[11]:


dict_of_indeces_with_data_and_labels_in_list = {}
for index in indeces:
#index = 'ndvi'


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
    

    
    data_tif_path = data_folder_path + str(filling_mode) + '_' + str(index) + '_raster.tif'    
    
    
    
    data_tif = gdal.Open(data_tif_path)
    data = np.array(data_tif.GetRasterBand(1).ReadAsArray())
    # plot.show(data)
#     print(np.unique(data))



#    #OPEN MULTIBAND DATA TIF AND SAVE THEM AS SEPERATE BAND TIF
    data = gdal.Open(data_tif_path)
    no_bands = data.RasterCount
    data_array = []
    data_tif = []
    for i in range(no_bands):
        data_tif.append(data.GetRasterBand(i+1))
        data_array.append(data.GetRasterBand(i+1).ReadAsArray())
    #     print(data_array[i].shape)
    #     plot.show(data_array[i])
        fu.array_to_raster(data_array[i],data,data_folder_path + separate_bands_folder_path + str(index) + '_date_' + str(i+1))

   #OPEN SEPERATE BAND TIFS AND CALCULATE STATISTICS USING SHAPEFILES
    stats_dicts_for_each_crop_and_date = []
    for shape in shapefiles_name_list:
#         print(shape)
        summed_file_path = shapefiles_folder_path + shape
#         print(summed_file_path)
        stats_of_each_date = []
        for i in range(len(data_tif)):
#             print(i)
            stats_of_each_date.append(rs.zonal_stats(summed_file_path, data_folder_path + separate_bands_folder_path +  str(index) + '_date_' + str(i+1), stats="count min mean max", nodata=-9999))
        stats_dicts_for_each_crop_and_date.append(stats_of_each_date)

    #SAVE THE STATISTICS AS CSVs
    for idx,crop in enumerate(crop_names_list):  
        for i in range(len(stats_dicts_for_each_crop_and_date[idx])):
            toCSV = stats_dicts_for_each_crop_and_date[idx][i]
            keys = toCSV[0].keys()
            with open(csv_path + crop_names_list[idx] + '_date_' + str(i+1) + str('_' + index) + '_stats.csv', 'w') as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(toCSV)

# #READ CSVs WITH STATISTICS, TAKE THE METRIC TO UTILIZE, CREATE DATA AND LABELS ARRAYS PER CROP
    index_means = []
    index_labels = []
    for j,crop in enumerate(crop_names_list):
        date_array = []
        for i in range(no_bands):
            stats = pd.read_csv(data_folder_path + 'CSVs/'+ crop + '_date_' + str(i+1) + str('_' + index) +'_stats.csv')
            date_array.append(stats.values)
    #    crop_arrays.append(np.asarray(date_array))
        temp = np.asarray(date_array)[:,:,2] #mean
        print(temp.shape)
        index_labels.append(np.ones(temp.shape[1])*labels_nums[j])
        index_means.append(np.transpose(temp))
        print(index_means[j].shape)
        
    #dict_of_indeces_with_data_and_labels_in_list[index] = [index_means, index_labels]
    data = np.copy(index_means[0])
    labels = np.copy(index_labels[0])
    for i in range(1,len(index_labels)):
        data = np.concatenate((data,index_means[i]),axis=0)
        labels = np.concatenate((labels,index_labels[i]),axis=0)


#print(len(dict_of_indeces_with_data_and_labels_in_list['ndvi'][1]))


#print(len(dict_of_indeces_with_data_and_labels_in_list['ndvi'][1]))

print(data.shape)
print(labels.shape)
#%%
#CREATE TRAIN TEST DATASETS FOR ML CLASSIFIERS
train_test_ratio = 0.2

clean_dataset = input('Clean Dataset with Neighbourhood Cleaning Rule? (y/n): ')

if clean_dataset == 'y':
    balanced = "balanced"
    file.write("UnderSampling: " + str(under_sampling_analysis) + "\n\n")
    total_elements = len(crops_only_flatten)
    background_elements = len(crops_only_flatten[crops_only_flatten != 0])
    resample_dict = {0: int(background_elements)}

    rus = NeighbourhoodCleaningRule(sampling_strategy='all')
    # rus = TomekLinks(sampling_strategy='all')
    # rus = RandomUnderSampler(sampling_strategy="not minority")
    # rus = OneSidedSelection(sampling_strategy='all',n_seeds_S=1000)

    start_train = time.time()
    X_rus, y_rus = rus.fit_sample(data_array_combined_flatten, crops_only_flatten)
    end_train = time.time()
    print("Balancing time: ", end_train - start_train)

    file.write("Balancing time: " + str(end_train - start_train) + "\n\n")

    for i in unique_labels:
        print("Before Class " + str(i) + " number of samples: ", len(crops_only_flatten[crops_only_flatten == i]))
        file.write("Before Class " + str(i) + " number of samples: " + str(
            len(crops_only_flatten[crops_only_flatten == i])) + "\n")
    file.write("\n")
    for i in unique_labels:
        print("After Class " + str(i) + " number of samples: ", len(y_rus[y_rus == i]))
        file.write("After Class " + str(i) + " number of samples: " + str(len(y_rus[y_rus == i])) + "\n")
    file.write("\n")

    # print(data_array_combined_flatten.shape)
    # print(crops_only_flatten.shape)

    # print(X_rus.shape)
    # print(y_rus.shape)

    X_train, X_test, y_train, y_test = train_test_split(X_rus, y_rus, stratify=y_rus, test_size=0.20, random_state=42)
    # print(X_train.shape)
    # print(y_train.shape)

else:

    X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(data, labels, test_size=train_test_ratio, random_state=42,stratify=labels)

#%%   28,14,7  200,100,50
mlp = MLPClassifier(hidden_layer_sizes=(200,100,50),max_iter=1000,activation='relu',random_state=42, verbose=True,batch_size=10)
start_train = time.time()
mlp.fit(X_train_ml,y_train_ml)
end_train = time.time()
print("MLP train time: ", end_train - start_train)
start_test = time.time()
y_pred_mlp = mlp.predict(X_test_ml)
end_test = time.time()
print("MLP test time: ", end_test - start_test)


print(mlp.loss_)


start_test = time.time()
y_train_pred_mlp = mlp.predict(X_train_ml)
end_test = time.time()
print("Train_accuracy test time: ", end_test - start_test)


cm = confusion_matrix(y_pred_mlp, y_test_ml)
train_cm = confusion_matrix(y_train_pred_mlp, y_train_ml)
print("Test Accuracy of MLPClassifier : ", fu.accuracy(cm))
print("Train Accuracy of MLPClassifier : ", fu.accuracy(train_cm))



stat_res = precision_recall_fscore_support(y_test_ml, y_pred_mlp,labels=labels_nums)
print(stat_res)
fu.print_confusion_matrix(cm,labels_nums)

# In[16]:


rf_obj = RandomForestClassifier(random_state=0)
start_train = time.time()
rf_obj.fit(X_train_ml, y_train_ml)
end_train = time.time()
print("RF train time: ",end_train - start_train)

start_test = time.time()
y_predict_rf = rf_obj.predict(X_test_ml)
end_test = time.time()
print("RF test time: ", end_test - start_test)

rf_y_pred_train_ml = rf_obj.predict(X_train_ml)


cm_rf = confusion_matrix(y_predict_rf, y_test_ml)
cm_train_rf = confusion_matrix(rf_y_pred_train_ml, y_train_ml)

print("Test Accuracy of RF Classifier : ", fu.accuracy(cm_rf))
print("Train Accuracy of RF Classifier : ", fu.accuracy(cm_train_rf))

stat_res_rf = precision_recall_fscore_support(y_test_ml, y_predict_rf,labels=labels_nums)
print(stat_res_rf)
fu.print_confusion_matrix(cm_rf,labels_nums)


rf_obj_importance = rf_obj.feature_importances_

print(rf_obj_importance)
# In[17]:


svm_clf = SVC(gamma='auto')
start_train = time.time()
svm_clf.fit(X_train_ml, y_train_ml)
end_train = time.time()
print("SVM train time: ",end_train - start_train)

start_test = time.time()
y_predict_svm = svm_clf.predict(X_test_ml)
end_test = time.time()
print("SVM test time: ", end_test - start_test)

svm_y_pred_train_ml = svm_clf.predict(X_train_ml)

cm_svm = confusion_matrix(y_predict_svm, y_test_ml)
cm_train_svm = confusion_matrix(svm_y_pred_train_ml, y_train_ml)


print("Test Accuracy of SVM Classifier : ", fu.accuracy(cm_svm))
print("Train Accuracy of SVM Classifier : ", fu.accuracy(cm_train_svm))


stat_res_svm = precision_recall_fscore_support(y_test_ml, y_predict_svm,labels=labels_nums)
print(stat_res_svm)
fu.print_confusion_matrix(cm_svm,labels_nums)






# In[18]:


neigh_num = 3
knn_clf = KNeighborsClassifier(n_neighbors=neigh_num)

start_train = time.time()
knn_clf.fit(X_train_ml, y_train_ml)
end_train = time.time()

print("KNN train time: ",end_train - start_train)

start_test = time.time()
y_predict_knn = knn_clf.predict(X_test_ml)
end_test = time.time()

print("KNN test time: ", end_test - start_test)

knn_y_pred_train_ml = knn_clf.predict(X_train_ml)

cm_knn = confusion_matrix(y_predict_knn, y_test_ml)
cm_train_knn = confusion_matrix(knn_y_pred_train_ml, y_train_ml)

print("Test Accuracy of KNN Classifier : ", fu.accuracy(cm_knn))
print("Train Accuracy of KNN Classifier : ", fu.accuracy(cm_train_knn))

stat_res_knn = precision_recall_fscore_support(y_test_ml, y_predict_knn,labels=labels_nums)
print(stat_res_knn)
fu.print_confusion_matrix(cm_knn,labels_nums)
# In[19]:


#TRAIN-TEST DATASETS FOR RMSE METHOD
#GETTING THE MEANS OF EACH CLASS AND USING RMSE WE CLASSIFY THE TEST DATA TO THE MINIMUM ERROR CLASS
X_test = []
y_test = []

train_test_ratio = 0.2


means = []
for i in range(len(index_labels)): 
    train_x, test_x, train_y, test_y = train_test_split(index_means[i],index_labels[i], test_size=train_test_ratio, random_state=42)
    
    means.append(np.mean(train_x, axis=0))
    
    X_test.append(test_x)
    y_test.append(test_y)


# print(means[0].shape)
# print(X_test[0].shape)
# print(means[1].shape)
# print(X_test[1].shape)
# print(means[2].shape)
# print(X_test[2].shape)


for line in range(len(means)):
    plt.plot(means[line])
plt.legend(crop_names_list)
plt.savefig('Lines.png')



test_data = np.copy(X_test[0])
test_labels = np.copy(y_test[0])
for i in range(1,len(y_test)):
    test_data = np.concatenate((test_data,X_test[i]),axis=0)
    test_labels = np.concatenate((test_labels,y_test[i]),axis=0)

# test_data = np.concatenate([X_test[0],X_test[1],X_test[2]])
# test_labels = np.concatenate([y_test[0],y_test[1],y_test[2]])

# print(test_data.shape)
# print(test_labels.shape)

results = []
for vec in test_data:
    rmse = []
    for crop_graph in means:
        # rmse.append((mean_absolute_error(crop_graph, vec)))        
        rmse.append((mean_squared_error(crop_graph, vec)))
    results.append(labels_nums[rmse.index(min(rmse))])
    
        

cm = confusion_matrix(test_labels,results )
stat_res = precision_recall_fscore_support(test_labels, results,labels=labels_nums)
print(stat_res)
fu.print_confusion_matrix(cm,labels_nums)
print("Test Accuracy of RMSE Classifier : ", fu.accuracy(cm))

# In[33]:


conf = confusion_matrix(test_labels, results)
stat_res = precision_recall_fscore_support(test_labels, results,labels=labels_nums, average='macro')
print(stat_res)



# In[37]:


stat_res = precision_recall_fscore_support(test_labels, results,labels=labels_nums)
print(stat_res)
fu.print_confusion_matrix(conf,crop_names_list)



