#!/usr/bin/env python3
# coding: utf-8
#

global os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

import os
import time
from osgeo import gdal
import csv

import gc


#custom files import
import pathlib

os.chdir(str(pathlib.Path(__file__).parent.absolute()))
test = pathlib.Path(__file__).parent.absolute()

#%%
import functions as fu
#import functions

workspace_path = str(pathlib.Path(pathlib.Path(__file__).parent.absolute()).parent)
print(workspace_path)

# year = '_2019/'
# year = '_2018/'
# area = ''

area_used = str(input('Give the number of the area to process (2,3,7,8): '))


year = '/'
area = '/area_' + area_used

data_folder_path = workspace_path + area + '/Data' + year
labels_folder_path = workspace_path + area + '/Ground_Truth_Data' + year
plots_folder_path = workspace_path + area + '/Plots' + year


# data_folder_path = workspace_path + year + 'Data/'
# labels_folder_path = workspace_path + year + 'Ground_Truth_Data/'
# landsat_dataset = 'Landsat8_dataset/'
# sentinel_dataset = 'Sentinel2_dataset/'

# cropped_data ='cropped_data/'
cropped_data = ''
landsat_use_bands = [4, 5, 12]
sentinel_use_bands = [4, 8, 'cloud_mask']
landsat8_name = 'landsat8_band_'
sentinel2_name = 'sentinel2_band_'

metrics_list = ["Precision", "Recall","F1 Score"]
mask_suffix = '_mask.tif'

#%%


#IMPORT LABELS
txt_name = str(input('Name of the file to write results: '))
file = open(txt_name,"w")
file.write(area[1:] + "\n\n")
#Read as TIF
#crop_data_tif = gdal.Open(labels_folder_path + 'CDL.tif')
# crop_mask_tif = gdal.Open(labels_folder_path + 'CMASK.tif')
crops_only_tif = gdal.Open(labels_folder_path + 'CDL_CROPS_ONLY.tif')
print(labels_folder_path)
#Read as Arrays
#crop_data = np.array(crop_data_tif.GetRasterBand(1).ReadAsArray())# crop_data_tif.read(1).astype('float32')
# crop_mask = np.array(crop_mask_tif.GetRasterBand(1).ReadAsArray()) #crop_mask_tif.read(1).astype('float32')
crops_only = np.array(crops_only_tif.GetRasterBand(1).ReadAsArray()) #crops_only_tif.read(1).astype('float32')


#read csv with crops type
labels = pd.read_csv(labels_folder_path + 'CDL_data.tif.vat.csv',skipinitialspace=True)
existing_labels = pd.read_csv(labels_folder_path + 'cdl_existing_labels.csv',skipinitialspace=True)


idx1 = (pd.Index(existing_labels['Value'])).union([0])
idx2 = np.unique(crops_only)

# print((idx1))
# print((idx2))

all_existing_labels = labels[labels['VALUE'].isin(idx1)]
crop_existing_labels = labels[labels['VALUE'].isin(idx2)]

# print(all_existing_labels)
# print(crop_existing_labels)



#

ex = existing_labels.loc[existing_labels['Value']<100]
test = ex.nlargest(10,'Count')
print('Crops with largest acreage')
print(test)
selected_crops = input('Select crops(names as you see them in Category,seperated with 1 space): ')
temp1 = ex.loc[ex['Category'].isin(selected_crops.split())]

crop_names = (temp1['Category'].values).tolist()
crop_id = (temp1['Value'].values).tolist()
print('Crops selected')
print(temp1)
file.write('Crops Selected \n')
file.write(str(temp1) + '\n\n')
# file.close()
selected_crops_array = np.zeros((crops_only.shape))
for value in crop_id:
    selected_crops_array = selected_crops_array + (crops_only==value)

    
    
    
selected_crops_array = np.where(selected_crops_array,crops_only,0)
# plt.figure(figsize=(50,20))
# plot.show(selected_crops_array)
print(selected_crops_array.shape)

#Maybe its best to fill all the image, considering that the opening/closing process may include cloudy pixels
#LEFT TO CHECK RESULTS TO CONFIRM


fill_only_crops_of_interest = True

if not(fill_only_crops_of_interest):
    selected_crops_array = crops_only


unique_labels = np.unique(selected_crops_array)
print(unique_labels)



with open(data_folder_path + 'crops_names_and_id.csv', 'w') as f:  # Just use 'w' mode in 3.x
    w = csv.writer(f,delimiter=',')
    level_counter = 0
    max_levels = len(crop_id)
    while level_counter < max_levels:
        w.writerow((crop_names[level_counter], crop_id[level_counter])) 
        level_counter = level_counter + 1 



#

#Finding the paths of the datasets' folders
datasets_list = os.listdir(data_folder_path)
for dataset in datasets_list:
    
    sentinel_dataset = data_folder_path + list(filter(lambda x: x.startswith('Sentinel2'), datasets_list))[0] + '/'
    landsat_dataset = data_folder_path + list(filter(lambda x: x.startswith('Landsat'), datasets_list))[0] + '/'
sentinel_dataset_list = os.listdir(sentinel_dataset)
landsat_dataset_list = os.listdir(landsat_dataset)

sentinel_dataset_list = list(map(( lambda x: x + '/'), sentinel_dataset_list))
landsat_dataset_list = list(map(( lambda x: x + '/'), landsat_dataset_list))

sentinel_dataset_list.sort()
landsat_dataset_list.sort()

print(sentinel_dataset_list)
print(landsat_dataset_list)
print(sentinel_dataset)
print(landsat_dataset)


# In[5]:

ndvi = []
filled_data = []
#Read satellite data and apply mask
landsat_raw_data = []
landsat_masked_data = []
# landsat_ndvi = []
# landsat_evi = []

sentinel_raw_data = []
sentinel_masked_data = []
# sentinel_filled_data =[]
# sentinel_ndvi = []
# sentinel_evi = []

filling_mode = input("Select filling method (spat/temp/mixed/none): ")
file.write("Filling method selected: " + filling_mode + "\n")

start_train = time.time()
#Read Raw Data and Apply Masks for each satellite data
for date in landsat_dataset_list:
    temp = fu.read_data_from_1_date(landsat_dataset + date + cropped_data, landsat_use_bands, landsat8_name)
    if filling_mode=='none':
        landsat_raw_data.append(temp)
    else:
        landsat_masked_data.append(fu.landsat_mask(temp[:-1,:,:], temp[-1]))
    del(temp)
    gc.collect()

for date in sentinel_dataset_list:
    print(date)
    temp = fu.read_data_from_1_date(sentinel_dataset + date + cropped_data, sentinel_use_bands, sentinel2_name)
    if filling_mode=='none':
        sentinel_raw_data.append(temp)
    else:
        sentinel_masked_data.append(fu.sentinel_mask(temp[:-1,:,:],temp[-1]))
    del(temp)
    gc.collect()
    
end_train = time.time() 
print("Read and Mask Data time: ", end_train - start_train)
file.write("Read and Mask Data time: " + str(end_train - start_train) + "\n")



start_train = time.time()
if filling_mode == 'spat':    
    print("Spatial Filling")
    landsat_filled_data =[]
    sentinel_filled_data =[]


    #Fill Masked Values with SPATIAL CALCULATIONS 
    for date in landsat_masked_data:
        temp = []
        for band in date:
            means,std = fu.calculate_means_of_classes_in_1_band(band, selected_crops_array)
            temp.append(fu.spatial_fill_missing_values_in_1_band(band, selected_crops_array, means, std))
        landsat_filled_data.append(np.array(temp))

    del(landsat_masked_data)
    gc.collect()
        
    for date in sentinel_masked_data:
        temp = []
        for band in date:
            means, std = fu.calculate_means_of_classes_in_1_band(band, selected_crops_array)
            temp.append(fu.spatial_fill_missing_values_in_1_band(band, selected_crops_array, means,std))
        sentinel_filled_data.append(np.array(temp))

    del(sentinel_masked_data)
    gc.collect()

    
    #sort the dates of both satellites
    all_data_dates_names = landsat_dataset_list + sentinel_dataset_list
    filled_data = landsat_filled_data + sentinel_filled_data
    print(all_data_dates_names)
    all_data_dates_names, filled_data = zip(*sorted(zip(all_data_dates_names,filled_data)))
    print(all_data_dates_names)

elif filling_mode=='temp' or filling_mode=='mixed':

    # Fill Masked Values with TEMPORAL CALCULATIONS
    print("Temporal Filling")
    #sort the dates of both satellites
    all_data_dates_names = landsat_dataset_list + sentinel_dataset_list
    filled_data = landsat_masked_data + sentinel_masked_data
    print(all_data_dates_names)
    all_data_dates_names, filled_data = zip(*sorted(zip(all_data_dates_names,filled_data)))
    print(all_data_dates_names)
    
    
    filled_data = fu.temporal_fill_missing_values(filled_data,selected_crops_array,filling_mode)
    print(filled_data.shape)
    del(landsat_masked_data)
    del(sentinel_masked_data)
    gc.collect()

else:
    print("No Filling")
    all_data_dates_names = landsat_dataset_list + sentinel_dataset_list
    filled_data = landsat_raw_data + sentinel_raw_data
    print(all_data_dates_names)
    all_data_dates_names, filled_data = zip(*sorted(zip(all_data_dates_names,filled_data)))
    print(all_data_dates_names)

    
    
end_train = time.time()
print("Fill Data time: ", end_train - start_train)
file.write("Fill Data time: " + str(end_train - start_train) + "\n")   

start_train = time.time()    
#Calculate Indeces NDVI
for date in filled_data:
    ndvi.append(fu.calculate_NDVI(date[0], date[1]))

end_train = time.time()
print("Calculate NDVI time: ", end_train - start_train)
file.write("Calculate NDVI time: " + str(end_train - start_train) + "\n\n")   

del(filled_data)
# del(ndvi)
# del(data_array_ndvi)
# del(landsat_masked_data)
# del(sentinel_masked_data)
# del(date)
# del(temp)
# del(all_masked_data_dates)
gc.collect()
# print("first garbage collection made")


# del(data_array_combined_flatten)
# del(crops_only_flatten)
# gc.collect()


#SAVE NDVI AND EVI RASTERS
ndvi = np.squeeze(np.array(ndvi))
# data_array_evi = np.squeeze(np.array(evi))
print(ndvi.shape)



fu.array_to_raster(ndvi,crops_only_tif,data_folder_path + str(filling_mode) + "_ndvi_raster.tif")



# array_to_raster(data_array_evi,crops_only_tif,data_folder_path + "evi_raster.tif")
print(ndvi.shape)
file.close()

# #%%
# for j in ndvi:
#     plt.figure(figsize=(50,20))
#     plt.imshow(j)
# #%%

"""

pixel_based_analysis = input("Conduct Pixel-Based Analysis? (y/n)? ")
if pixel_based_analysis=='n':
    file.close()
    sys.exit(0)



# In[27]:


#COMBINE NDVI AND EVI INDECES TO ONE ARRAY

print(len(data_array_ndvi))
number_of_bands_combined = len(data_array_ndvi)
print(data_array_ndvi.shape)

data_array_combined_flatten = np.transpose(data_array_ndvi.reshape((number_of_bands_combined,-1)))
crops_only_flatten = selected_crops_array.reshape((-1))
print(data_array_combined_flatten.shape)
print(crops_only_flatten.shape)

x, y = data_array_combined_flatten.shape
print(x,y)



#%%

under_sampling_analysis = input("Conduct UnderSampling Analysis? (yes/no)? ")
if under_sampling_analysis=='y':
    
    balanced = "balanced"
    file.write("UnderSampling: " + str(under_sampling_analysis) + "\n\n")
    total_elements = len(crops_only_flatten)
    background_elements = len(crops_only_flatten[crops_only_flatten!=0])
    resample_dict = {0: int(background_elements)}
    
    rus = NeighbourhoodCleaningRule(sampling_strategy='all')
    # rus = TomekLinks(sampling_strategy='all')
    # rus = RandomUnderSampler(sampling_strategy="not minority")
    # rus = OneSidedSelection(sampling_strategy='all',n_seeds_S=1000)
    
    # for i in unique_labels:
    #     print("Before Class" + str(i) +  "number of samples: ", len(crops_only_flatten[crops_only_flatten==i]))
    #     file.write("Before Class" + str(i) +  "number of samples: " + str(len(crops_only_flatten[crops_only_flatten==i])) + "\n")
    
    # print("Before Class 0 number of samples: ", len(crops_only_flatten[crops_only_flatten==0]))
    # print("Before Class 1 number of samples: ", len(crops_only_flatten[crops_only_flatten==1]))
    # print("Before Class 2 number of samples: ", len(crops_only_flatten[crops_only_flatten==2]))
    # print("Before Class 10 number of samples: ", len(crops_only_flatten[crops_only_flatten==10]))
    # print()
    start_train = time.time()
    X_rus, y_rus = rus.fit_sample(data_array_combined_flatten, crops_only_flatten)
    end_train = time.time()
    print("Balancing time: ", end_train - start_train)
    # print()
    # print("After Class 0 number of samples: ", len(y_rus[y_rus==0]))
    # print("After Class 1 number of samples: ", len(y_rus[y_rus==1]))
    # print("After Class 2 number of samples: ", len(y_rus[y_rus==2]))
    # print("After Class 10 number of samples: ", len(y_rus[y_rus==10]))
    # print()
    
    file.write("Balancing time: " + str(end_train - start_train) + "\n\n")
    
    for i in unique_labels:
        print("Before Class " + str(i) +  " number of samples: ", len(crops_only_flatten[crops_only_flatten==i]))
        file.write("Before Class " + str(i) +  " number of samples: " + str(len(crops_only_flatten[crops_only_flatten==i])) + "\n")
    file.write("\n")
    for i in unique_labels:
        print("After Class " + str(i) +  " number of samples: ", len(y_rus[y_rus==i]))
        file.write("After Class " + str(i) +  " number of samples: " + str(len(y_rus[y_rus==i])) + "\n")
    file.write("\n")
    
    
    # print(data_array_combined_flatten.shape)
    # print(crops_only_flatten.shape)
    
    # print(X_rus.shape)
    # print(y_rus.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X_rus, y_rus,stratify=y_rus, test_size=0.20, random_state=42)
    # print(X_train.shape)
    # print(y_train.shape)
else: 
    file.write("No UnderSampling \n\n")
    balanced = ""
    print("No UnderSampling Train-Test splitting")
    # X_train, X_test, y_train, y_test = train_test_split(X_rus, y_rus,stratify=y_rus, test_size=0.20, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(data_array_combined_flatten, crops_only_flatten,stratify=crops_only_flatten, test_size=0.20, random_state=42)
    
#%%
# del(filled_data)
# del(ndvi)
# del(data_array_ndvi)
# del(landsat_masked_data)
# del(sentinel_masked_data)
# del(date)
# del(temp)
# del(all_masked_data_dates)
# gc.collect()
# print("first garbage collection made")


# del(data_array_combined_flatten)
# del(crops_only_flatten)
# gc.collect()

#%%
print("MLP Running")
#SPAT
#200/100/50 loss=0.07352, 0.9649,0.9767, f1 > 0.88219, b=400 1:30 hours approximately
#200/100/50 loss=0.08871, 0.9645,0.9704, f1 > 0.87899, b=5000 33.5 minutes
#200/100/50 loss=0.07623, 0.9654,0.9770, f1 > 0.88397, b=200 1:10 hours
#150/100/50 loss=0.07885, 0.9643,0.9744, f1 > 0.87793, b=400 1:10 hours

#TEMP
#200/100/50 loss=0.09302, 0.9501,0.9676, f1 > 0.80629 , b=400 2:20 hours approximately
#200/100/50 loss=0.02112, 0.9864,0.9931, f1 > 0.94223
mlp_clf = MLPClassifier(hidden_layer_sizes=(200,100,50),max_iter=500,activation='relu',random_state=42, verbose=True,batch_size=400)
start_train = time.time()
mlp_clf.fit(X_train,y_train)
end_train = time.time()
print("MLP train time: ", end_train - start_train)
file.write("MLP train time: " + str(end_train - start_train) + "\n")

start_test = time.time()
y_pred = mlp_clf.predict(X_test)
end_test = time.time()
print("MLP test time: ", end_test - start_test)

file.write("MLP test time: " + str(end_test - start_test) + "\n")
file.write("MLP Final loss: " + str(mlp_clf.loss_) + "\n")
print(mlp_clf.loss_)



start_test = time.time()
y_train_pred = mlp_clf.predict(X_train)
end_test = time.time()
print("Train_accuracy test time: ", end_test - start_test)

cm = confusion_matrix(y_pred, y_test)
train_cm = confusion_matrix(y_train_pred, y_train)
print("MLP Classifier Test Accuracy: ", fu.accuracy(cm))
print("MLP Classifier Train Accuracy: ", fu.accuracy(train_cm))
file.write("MLP Classifier Test Accuracy: " + str(fu.accuracy(cm)) + "\n")
file.write("MLP Classifier Train Accuracy: " + str(fu.accuracy(train_cm)) + "\n")


stat_res = precision_recall_fscore_support(y_test, y_pred,labels=unique_labels)
print(stat_res)
file.write("MLP Precision, Recall, F1 score \n\n")
for i,met in enumerate(metrics_list):
    file.write(met + "\n")
    file.write(str(stat_res[i]) + "\n")

fu.print_confusion_matrix(cm,np.unique(selected_crops_array))
file.write("\n")

plt.savefig(plots_folder_path + "MLP_Conf_Matrix_PBIA_" + filling_mode + "_" + balanced + ".png")
# In[24]:
#TEMP
# 0.95308,0.99918, f1 > 0.824845, 11 mins 'gini' n=100
# 0.95323,0.99918, f1 > 0.823428, 14 mins 'entropy' n=100

rf = RandomForestClassifier(random_state=0,n_estimators=100,n_jobs=-1)#,criterion='entropy')
start_train = time.time()
rf.fit(X_train, y_train)
end_train = time.time()
print("RF train time: ",end_train - start_train)
file.write("RF train time: " + str(end_train - start_train) + "\n")

start_test = time.time()
rf_y_pred = rf.predict(X_test)
end_test = time.time()
print("RF test time: ", end_test - start_test)
file.write("RF test time: " + str(end_test - start_test) + "\n")

start_test = time.time()
rf_y_pred_train = rf.predict(X_train)
end_test = time.time()


rf_cm = confusion_matrix(rf_y_pred, y_test)
rf_cm_train = confusion_matrix(rf_y_pred_train, y_train)
rf_stat_res = precision_recall_fscore_support(y_test, rf_y_pred,labels=unique_labels)


print("RF Classifier Test Accuracy: ", fu.accuracy(rf_cm))
print("RF Classifier Train Accuracy: ", fu.accuracy(rf_cm_train))
file.write("RF Classifier Test Accuracy: " + str(fu.accuracy(rf_cm)) + "\n")
file.write("RF Classifier Train Accuracy: " + str(fu.accuracy(rf_cm_train)) + "\n")
print(rf_stat_res)

file.write("\nRF Precision, Recall, F1 score \n\n")

for i,met in enumerate(metrics_list):
    file.write(met + "\n")
    file.write(str(rf_stat_res[i]) + "\n")

fu.print_confusion_matrix(rf_cm,unique_labels)

plt.savefig(plots_folder_path+ "RF_Conf_Matrix_PBIA_" + filling_mode + "_" + balanced + ".png")

opt = np.get_printoptions()
np.set_printoptions(threshold=np.inf)
importance = rf.feature_importances_
print((importance))
np.set_printoptions(**opt)
file.write("RF importances: " + str(importance))

file.close()
# In[37]:

# disp = plot_confusion_matrix(rf,X_test,y_test,display_labels=unique_labels,cmap=plt.cm.jet,normalize='true')
# print(disp.confusion_matrix)
# plt.show()
# print(rf.classes_)
# print(rf.n_classes_)
# print(rf.n_outputs_)
# print(rf.n_features_)
# print(rf.feature_importances_)
# print((np.unique(crops_only)))
# print()


# In[39]:


# temp = [estimator.tree_.max_depth for estimator in rf.estimators_]
# print(len(temp))
# print(np.mean(np.array(temp)))
# print(np.amax(np.array(temp)))
# print(np.amin(np.array(temp)))


"""
