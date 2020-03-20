#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
# import rasterio
# from rasterio import plot
import pandas as pd
import sys
global os
import os
import time
import gdal
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC, LinearSVR
# from sklearn.neighbors import KNeighborsClassifier

#custom files import
import pathlib
os.chdir(pathlib.Path(__file__).parent.absolute())

import functions as fu
#import functions

workspace_path = str(pathlib.Path(pathlib.Path(__file__).parent.absolute()).parent)
print(workspace_path)

year = '_2018/'
# year = '_2019/'
data_folder_path = workspace_path + '/Data' + year
labels_folder_path = workspace_path + '/Ground_Truth_Data' + year
# landsat_dataset = 'Landsat8_dataset/'
# sentinel_dataset = 'Sentinel2_dataset/'

cropped_data ='cropped_data/'

landsat_use_bands = [4, 5, 12]
sentinel_use_bands = [4, 8, 'cloud_mask']
landsat8_name = 'landsat8_band_'
sentinel2_name = 'sentinel2_band_'


mask_suffix = '_mask.tif'


# In[2]:


#IMPORT LABELS


#Read as TIF
#crop_data_tif = gdal.Open(labels_folder_path + 'CDL.tif')
# crop_mask_tif = gdal.Open(labels_folder_path + 'CMASK.tif')
crops_only_tif = gdal.Open(labels_folder_path + 'CDL_CROPS_ONLY.tif')

#Read as Arrays
#crop_data = np.array(crop_data_tif.GetRasterBand(1).ReadAsArray())# crop_data_tif.read(1).astype('float64')
# crop_mask = np.array(crop_mask_tif.GetRasterBand(1).ReadAsArray()) #crop_mask_tif.read(1).astype('float64')
crops_only = np.array(crops_only_tif.GetRasterBand(1).ReadAsArray()) #crops_only_tif.read(1).astype('float64')


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


# In[3]:


ex = existing_labels.loc[existing_labels['Value']<100]
test = ex.nlargest(10,'Count')
print('Crops with largest acreage')
print(test)
selected_crops = input('Select crops(names as you see,seperated with 1 space): ')
temp1 = ex.loc[ex['Category'].isin(selected_crops.split())]

crop_id = (temp1['Value'].values).tolist()
print('Crops selected')
print(temp1)

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

print(np.unique(selected_crops_array))


# In[4]:


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
# landsat_raw_data = []
landsat_masked_data = []
# landsat_ndvi = []
# landsat_evi = []

# sentinel_raw_data = []
sentinel_masked_data = []
# sentinel_filled_data =[]
# sentinel_ndvi = []
# sentinel_evi = []

start_train = time.time()
#Read Raw Data and Apply Masks for each satellite data
for date in landsat_dataset_list:
    temp = fu.read_data_from_1_date(landsat_dataset + date + cropped_data, landsat_use_bands, landsat8_name)
#     landsat_raw_data.append(temp)
    landsat_masked_data.append(fu.landsat_mask(temp[:-1,:,:], temp[-1]))

for date in sentinel_dataset_list:
    temp = fu.read_data_from_1_date(sentinel_dataset + date + cropped_data, sentinel_use_bands, sentinel2_name)
#     sentinel_raw_data.append(temp)
    sentinel_masked_data.append(fu.sentinel_mask(temp[:-1,:,:],temp[-1]))

end_train = time.time() 
print("Read And mask Data time: ", end_train - start_train)


#Fill Masked Values with TEMPORAL CALCULATIONS 
filling_mode = input("Select filling method (spat/temp): ")
start_train = time.time()
if filling_mode == 'spat':    

    landsat_filled_data =[]
    sentinel_filled_data =[]


    #Fill Masked Values with SPATIAL CALCULATIONS 
    for date in landsat_masked_data:
        temp = []
        for band in date:
            means,std = fu.calculate_means_of_classes_in_1_band(band, selected_crops_array)
            temp.append(fu.spatial_fill_missing_values_in_1_band(band, selected_crops_array, means, std))
        landsat_filled_data.append(np.array(temp))
        
    for date in sentinel_masked_data:
        temp = []
        for band in date:
            means, std = fu.calculate_means_of_classes_in_1_band(band, selected_crops_array)
            temp.append(fu.spatial_fill_missing_values_in_1_band(band, selected_crops_array, means,std))
        sentinel_filled_data.append(np.array(temp))

    
    #sort the dates of both satellites
    all_data_dates_names = landsat_dataset_list + sentinel_dataset_list
    all_masked_data_dates = landsat_filled_data + sentinel_filled_data
    print(all_data_dates_names)
    all_data_dates_names, filled_data = zip(*sorted(zip(all_data_dates_names,all_masked_data_dates)))
    print(all_data_dates_names)

else:
    #sort the dates of both satellites
    all_data_dates_names = landsat_dataset_list + sentinel_dataset_list
    all_masked_data_dates = landsat_masked_data + sentinel_masked_data
    print(all_data_dates_names)
    all_data_dates_names, all_masked_data_dates = zip(*sorted(zip(all_data_dates_names,all_masked_data_dates)))
    print(all_data_dates_names)
    
    
    filled_data = fu.temporal_fill_missing_values(all_masked_data_dates,selected_crops_array)
    
    
end_train = time.time()
print("Fill Data time: ", end_train - start_train)   

start_train = time.time()    
#Calculate Indeces NDVI, EVI
for date in filled_data:
    ndvi.append(fu.calculate_NDVI(date[0], date[1]))
#     landsat_evi.append(fu.calculate_EVI(date[0], date[1], date[2]))

# for date in sentinel_filled_data:
#     sentinel_ndvi.append(fu.calculate_NDVI(date[0], date[1]))
# #     sentinel_evi.append(fu.calculate_EVI(date[0], date[1], date[2]))
end_train = time.time()
print("Calculate NDVI time: ", end_train - start_train)


# In[7]:


# plt.figure(figsize=(50,20))
# plt.imshow(landsat_raw_data[4][0])
plt.figure(figsize=(50,20))
plt.imshow(landsat_masked_data[4][0])
# plt.figure(figsize=(50,20))
# plt.imshow(selected_crops_array)
plt.figure(figsize=(50,20))
plt.imshow(landsat_filled_data[4][0])
# plt.figure(figsize=(50,20))
# plt.imshow(sentinel_raw_data[6][0])
plt.figure(figsize=(50,20))
plt.imshow(sentinel_masked_data[6][0])
plt.figure(figsize=(50,20))
plt.imshow(sentinel_filled_data[6][0])
plt.show()



# In[8]:


#SAVE NDVI AND EVI RASTERS
data_array_ndvi = np.squeeze(np.array(ndvi))
# data_array_evi = np.squeeze(np.array(evi))
print(data_array_ndvi.shape)
fu.array_to_raster(data_array_ndvi,crops_only_tif,data_folder_path + "ndvi_raster.tif")
# array_to_raster(data_array_evi,crops_only_tif,data_folder_path + "evi_raster.tif")
print(data_array_ndvi.shape)


# In[3]:


pixel_based_analysis = input("Conduct Pixel-Based Analisys? (y/n)? ")
if pixel_based_analysis=='n':
    sys.exit(0)


# In[27]:


#COMBINE NDVI AND EVI INDECES TO ONE ARRAY
data_array_combined = ndvi

print(len(data_array_combined))
number_of_bands_combined = len(data_array_combined)
print(data_array_combined.shape)

data_array_combined_flatten = np.transpose(data_array_combined.reshape((number_of_bands_combined,-1)))
crops_only_flatten = crops_only.reshape((-1))
print(data_array_combined_flatten.shape)
print(crops_only_flatten.shape)

x, y = data_array_combined_flatten.shape
print(x,y)
X_train, X_test, y_train, y_test = train_test_split(data_array_combined_flatten, crops_only_flatten, test_size=0.20, random_state=42)
print(X_train.shape)
print(y_train.shape)


# In[24]:


# clf = RandomForestClassifier(random_state=0)
# start_train = time.time()
# clf.fit(X_train, y_train)
# end_train = time.time()
# print(end_train - start_train)

# start_test = time.time()
# result = clf.score(X_test,y_test)
# end_test = time.time()
# print(end_test - start_test)

# print(result)


# In[28]:


# clf = RandomForestClassifier(random_state=0)
# start_train = time.time()
# clf.fit(X_train, y_train)
# end_train = time.time()
# print(end_train - start_train)

# start_test = time.time()
# result = clf.score(X_test,y_test)
# end_test = time.time()
# print(end_test - start_test)

# print(result)


# In[37]:


# print(clf.classes_)
# print(clf.n_classes_)
# print(clf.n_outputs_)
# print(clf.n_features_)
# print(clf.feature_importances_)
# print((np.unique(crops_only)))
# print()


# In[39]:


# temp = [estimator.tree_.max_depth for estimator in clf.estimators_]
# print(len(temp))
# print(np.mean(np.array(temp)))
# print(np.amax(np.array(temp)))
# print(np.amin(np.array(temp)))



