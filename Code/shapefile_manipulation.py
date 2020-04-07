#!/usr/bin/env python
# coding: utf-8

# In[26]:


import matplotlib.pyplot as plt
import numpy as np
from rasterio import plot
import pandas as pd
import os
import gdal
import csv
import time




from math import sqrt

import rasterstats as rs
import functions as fu


from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier



# In[10]:


# year = '_2018/'
year = '_2019/'
data_folder_path = '/home/chris/Desktop/diploma/Diploma-Workspace/Data' + year
labels_folder_path = '/home/chris/Desktop/diploma/Diploma-Workspace/Ground_Truth_Data' + year
shapefiles_folder_path = data_folder_path + 'Shapefiles/'
separate_bands_folder_path = 'Separate_Bands/'
csv_path = data_folder_path + 'CSVs/'


selected_indeces = input('Write Indeces(seperated with 1 space, options: ndvi, evi): ')
indeces = selected_indeces.split()
print(indeces)


# if 'ndvi' in indeces:
#     ndvi_data_tif_path = data_folder_path + 'ndvi_raster.tif'
#     ndvi_data_tif = gdal.Open(ndvi_data_tif_path)
#     ndvi_data = np.array(ndvi_data_tif.GetRasterBand(1).ReadAsArray())
#     plot.show(ndvi_data)
#     print(np.unique(ndvi_data))
    
# if 'evi' in indeces:
#     evi_data_tif_path = data_folder_path + 'evi_raster.tif'
#     evi_data_tif = gdal.Open(evi_data_tif_path)
#     evi_data = np.array(evi_data_tif.GetRasterBand(1).ReadAsArray())
#     plot.show(evi_data)
#     print(np.unique(evi_data))


# In[3]:


shapefiles_name_list = os.listdir(data_folder_path + 'Shapefiles/')
print((shapefiles_name_list))
threshold = input("Enter threshold of number of pixels per parcel for processing: ")
shapefiles_name_list = sorted([file for file in shapefiles_name_list if file.endswith(threshold + '.shp')])

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

    data_tif_path = data_folder_path + str(index) + '_raster.tif'
    data_tif = gdal.Open(data_tif_path)
    data = np.array(data_tif.GetRasterBand(1).ReadAsArray())
    plot.show(data)
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
#     stats_dicts_for_each_crop_and_date = []
#     for shape in shapefiles_name_list:
# #         print(shape)
#         summed_file_path = shapefiles_folder_path + shape
# #         print(summed_file_path)
#         stats_of_each_date = []
#         for i in range(len(data_tif)):
# #             print(i)
#             stats_of_each_date.append(rs.zonal_stats(summed_file_path, data_folder_path + separate_bands_folder_path +  str(index) + '_date_' + str(i+1), stats="count min mean max", nodata=-9999))
#         stats_dicts_for_each_crop_and_date.append(stats_of_each_date)

#     #SAVE THE STATISTICS AS CSVs
#     for idx,crop in enumerate(crop_names_list):  
#         for i in range(len(stats_dicts_for_each_crop_and_date[idx])):
#             toCSV = stats_dicts_for_each_crop_and_date[idx][i]
#             keys = toCSV[0].keys()
#             with open(csv_path + crop_names_list[idx] + '_date_' + str(i+1) + str('_' + index) + '_stats.csv', 'w') as output_file:
#                 dict_writer = csv.DictWriter(output_file, keys)
#                 dict_writer.writeheader()
#                 dict_writer.writerows(toCSV)

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
train_test_ratio = 0.5
  
    
X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(data, labels, test_size=train_test_ratio, random_state=42,stratify=labels)


mlp = MLPClassifier(hidden_layer_sizes=(28,14,7),max_iter=500,activation='relu',random_state=42, verbose=False,batch_size=50)
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
print(end_train - start_train)

start_test = time.time()
y_predict_rf = rf_obj.predict(X_test_ml)
end_test = time.time()
print(end_test - start_test)

cm_rf = confusion_matrix(y_predict_rf, y_test_ml)
stat_res = precision_recall_fscore_support(y_test_ml, y_predict_rf,labels=labels_nums)
print(stat_res)
fu.print_confusion_matrix(cm_rf,labels_nums)
print("Test Accuracy of RF Classifier : ", fu.accuracy(cm_rf))

rf_obj_importance = rf_obj.feature_importances_

print(rf_obj_importance)
# In[17]:


clf = SVC(gamma='auto')
start_train = time.time()
clf.fit(X_train_ml, y_train_ml)
end_train = time.time()
print(end_train - start_train)

start_test = time.time()
y_predict_svm = clf.predict(X_test_ml)
end_test = time.time()
print(end_test - start_test)

cm_svm = confusion_matrix(y_predict_svm, y_test_ml)
stat_res_svm = precision_recall_fscore_support(y_test_ml, y_predict_svm,labels=labels_nums)
print(stat_res)
fu.print_confusion_matrix(cm_svm,labels_nums)
print("Test Accuracy of SVM Classifier : ", fu.accuracy(cm_svm))


# In[18]:


neigh_num = 4
neigh = KNeighborsClassifier(n_neighbors=neigh_num)

start_train = time.time()
neigh.fit(X_train, y_train)
end_train = time.time()

print(end_train - start_train)

start_test = time.time()
result = neigh.score(X_test,y_test)
end_test = time.time()

print(end_test - start_test)

print(result)


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


# for line in range(len(means)):
#     plt.plot(means[line])
# plt.legend(crop_names_list)
# plt.savefig('Lines.png')



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
    
    
       
diff = results - test_labels
accuracy = len(diff[diff==0])/len(diff)
print(accuracy)    


# In[33]:


conf = confusion_matrix(test_labels, results)
stat_res = precision_recall_fscore_support(test_labels, results,labels=labels_nums, average='macro')
print(stat_res)



# In[37]:


stat_res = precision_recall_fscore_support(test_labels, results,labels=labels_nums)
print(stat_res)
fu.print_confusion_matrix(conf,crop_names_list)



