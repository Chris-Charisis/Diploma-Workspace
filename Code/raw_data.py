#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio import plot
import pandas as pd
import sys
import os
import time
import gdal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVR
from sklearn.neighbors import KNeighborsClassifier

#custom files import

from skimage import *


import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
import skimage.io as io


# from quality_assessment import *


data_folder_path = '/home/chris/Desktop/diploma/Diploma-Workspace/Data/'
labels_folder_path = '/home/chris/Desktop/diploma/Diploma-Workspace/Ground_Truth_Data/'
# landsat_dataset = 'Landsat8_dataset/'
# sentinel_dataset = 'Sentinel2_dataset/'

cropped_data ='cropped_data/'

landsat_use_bands = [2, 4, 5, 12]
sentinel_use_bands = [2, 4, 8, 'TCI', 'cloud_mask']
landsat8_name = 'landsat8_band_'
sentinel2_name = 'sentinel2_band_'


crop_name = ['Cotton', 'Corn', 'Peanuts']
mask_suffix = '_mask.tif'


# In[143]:


#IMPORT LABELS


#Read as TIF
crop_data_tif = gdal.Open(labels_folder_path + 'CDL.tif')
# crop_mask_tif = gdal.Open(labels_folder_path + 'CMASK.tif')
crops_only_tif = gdal.Open(labels_folder_path + 'CDL_CROPS_ONLY.tif')

#Read as Arrays
crop_data = np.array(crop_data_tif.GetRasterBand(1).ReadAsArray())# crop_data_tif.read(1).astype('float64')
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


# In[171]:


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


# In[180]:


def array_to_raster(array, old_raster_used_for_projection, save_path):

    width = old_raster_used_for_projection.RasterYSize
    height = old_raster_used_for_projection.RasterXSize
    gt = old_raster_used_for_projection.GetGeoTransform()
    wkt_projection = old_raster_used_for_projection.GetProjectionRef()

    if len(array.shape)==2:
        array = np.expand_dims(array, axis=0)

    no_bands =  array.shape[0]
    
    driver = gdal.GetDriverByName('GTiff')
    print(driver)
    DataSet = driver.Create(save_path, height, width, no_bands, gdal.GDT_Float64)
    DataSet.SetGeoTransform(gt)
    DataSet.SetProjection(wkt_projection)

    for i in range(no_bands):
        DataSet.GetRasterBand(i+1).WriteArray(array[i])
    DataSet = None
    
    return save_path


# In[191]:


#IMPORT SATELLITE DATA
def read_data_from_1_date(dataset, use_bands, satellite_name):
    #data =[]
    data_list = []
    files = []

    number_of_files = len(os.listdir(dataset))
    for i in use_bands:
        temp = rasterio.open(dataset + cropped_data + satellite_name + str(i) + '.tif')
        #data.append(temp)
        data_list.append(temp.read(1).astype('float64'))

                  
#         temp = gdal.Open(dataset + cropped_data + satellite_name + str(i) + '.tif')
#         #data.append(temp)
#         data_list.append(np.array(temp.GetRasterBand(1).ReadAsArray()))  
    number_of_bands = len(data_list)
    print(number_of_bands)
    data_array = np.array(data_list)
    return data_array


# In[182]:


#NDVI
def calculate_NDVI(red, nir):

    ndvi = np.where((nir+red)==0., 0, (nir-red)/(nir+red))
    ndvi = ndvi[np.newaxis,:,:]
    return ndvi


# In[183]:


#EVI
def calculate_EVI(blue, red, nir):
    evi = np.where((nir + 6*red - 7.5*blue + 1)==0., 0, 2.5*(nir-red)/(nir + 6*red - 7.5*blue + 1))
    evi = evi[np.newaxis,:,:]
    return evi


# In[184]:


def calculate_means_of_classes_in_1_band(data, labels):
    dif_labels = np.unique(labels)
    means = []
    std = []
    no_data = -9999
    for label in dif_labels:
        labeled_pixels = data[labels==label]
        labeled_pixels_without_no_value = labeled_pixels[labeled_pixels!=no_data]

        denominator = len(labeled_pixels_without_no_value)
        if denominator == 0: denominator = 1
        means.append(np.sum(labeled_pixels_without_no_value)/denominator)
        std.append(np.std(labeled_pixels_without_no_value))
    return means, std


# In[185]:


def fill_missing_values_in_1_band(data,labels,means,std):
    
    dif_labels = np.unique(labels)
    no_data = -9999
    no_data_values = data==no_data
    for idx,label in enumerate(dif_labels):
        label_values = labels==label
        values_to_fill = np.logical_and(no_data_values, label_values)
        
        norm_distr_values = np.random.normal(means[idx],std[idx],data.size)
        data = np.where(values_to_fill, means[idx], data)
    return data


# In[186]:


def landsat_mask(data,quality_band):
    # 2720 - CLEAR
    # 2752 - CLOUD CONFIDENCE MEDIUM
    # 2800 - CLOUD
    # 2976 - CLOUD SHADOW HIGH
    # 3008 - CIRRUS CONFIDENCE LOW
    
    #Valid Data Values 2720, 2724, 2728, 2732
    data_values = np.array([2720, 2724, 2728, 2732])
    
    mask = np.where(np.isin(quality_band,data_values), 1, -1)  
    
    num_of_bands = np.size(data, axis=0)
    for i in range(num_of_bands):
        data[i] = np.multiply(data[i], mask)
    
    return np.where(data<0, -9999, data)


# In[187]:


def sentinel_mask(data, mask):
    #Five 0 => clear land pixel
    #1 => clear water pixel
    #2 => cloud shadow
    #3 => snow
    #4 => cloud
    #255 => no observation
    mask = np.where(mask<2, 1, -1)
    
    num_of_bands = np.size(data, axis=0)
    for i in range(num_of_bands):
        data[i] = np.multiply(data[i], mask)
    
  
    return np.where(data<0, -9999, data)


# In[188]:


#Finding the paths of the datasets' folders
datasets_list = os.listdir(data_folder_path)
for dataset in datasets_list:
    
    sentinel_dataset = data_folder_path + list(filter(lambda x: x.startswith('Sentinel'), datasets_list))[0] + '/'
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


# In[192]:


#Read satellite data and apply mask
landsat_raw_data = []
landsat_masked_data = []
landsat_filled_data =[]
landsat_ndvi = []
landsat_evi = []

sentinel_raw_data = []
sentinel_masked_data = []
sentinel_filled_data =[]
sentinel_ndvi = []
sentinel_evi = []

#Read Raw Data and Apply Masks
for date in landsat_dataset_list:
    temp = read_data_from_1_date(landsat_dataset + date, landsat_use_bands, landsat8_name)
    landsat_raw_data.append(temp)
    landsat_masked_data.append(landsat_mask(temp[:-1,:,:], temp[-1]))

for date in sentinel_dataset_list:
    temp = read_data_from_1_date(sentinel_dataset + date, sentinel_use_bands, sentinel2_name)
    sentinel_raw_data.append(temp)
    sentinel_masked_data.append(sentinel_mask(temp[:-1,:,:],temp[-1]))

#Fill Masked Values
for date in landsat_masked_data:
    temp = []
    for band in date:
        means,std = calculate_means_of_classes_in_1_band(band, selected_crops_array)
        temp.append(fill_missing_values_in_1_band(band, selected_crops_array, means, std))
    landsat_filled_data.append(np.array(temp))
    
for date in sentinel_masked_data:
    temp = []
    for band in date:
        means, std = calculate_means_of_classes_in_1_band(band, selected_crops_array)
        temp.append(fill_missing_values_in_1_band(band, selected_crops_array, means,std))
    sentinel_filled_data.append(np.array(temp))
    
    
#Calculate Indeces NDVI, EVI
for date in landsat_filled_data:
    landsat_ndvi.append(calculate_NDVI(date[1], date[2]))
    landsat_evi.append(calculate_EVI(date[0], date[1], date[2]))

for date in sentinel_filled_data:
    sentinel_ndvi.append(calculate_NDVI(date[1], date[2]))
    sentinel_evi.append(calculate_EVI(date[0], date[1], date[2]))


# In[193]:


plt.figure(figsize=(50,20))
plt.imshow(landsat_raw_data[4][0])
plt.figure(figsize=(50,20))
plt.imshow(landsat_masked_data[4][0])
plt.figure(figsize=(50,20))
plt.imshow(selected_crops_array)
plt.figure(figsize=(50,20))
plt.imshow(landsat_filled_data[4][0])
plt.figure(figsize=(50,20))
plt.imshow(sentinel_raw_data[6][0])
plt.figure(figsize=(50,20))
plt.imshow(sentinel_masked_data[6][0])
plt.figure(figsize=(50,20))
plt.imshow(sentinel_filled_data[6][0])
plt.show()


# In[23]:


landsat_number_of_bands_ndvi = len(landsat_ndvi)
landsat_number_of_bands_evi = len(landsat_evi)
sentinel_number_of_bands_ndvi = len(sentinel_ndvi)
sentinel_number_of_bands_evi = len(sentinel_evi)

print(landsat_ndvi[0].shape)
print(sentinel_ndvi[0].shape)
print(landsat_number_of_bands_ndvi)
print(landsat_number_of_bands_evi)
print(sentinel_number_of_bands_ndvi)
print(sentinel_number_of_bands_evi)


ndvi = landsat_ndvi + sentinel_ndvi
evi = landsat_evi + sentinel_evi


# In[165]:


#COMBINE AND SORT DATA WITH TEMPORAL ORDER
all_data_dates = landsat_dataset_list + sentinel_dataset_list
print(all_data_dates)
all_data_dates, ndvi = zip(*sorted(zip(all_data_dates, ndvi)))
all_data_dates, evi = zip(*sorted(zip(all_data_dates, evi)))
print(all_data_dates)


# In[26]:


#SAVE NDVI AND EVI RASTERS
data_array_ndvi = np.squeeze(np.array(ndvi))
data_array_evi = np.squeeze(np.array(evi))

array_to_raster(data_array_ndvi,crops_only_tif,data_folder_path + "NDVI_raster.tif")
array_to_raster(data_array_evi,crops_only_tif,data_folder_path + "EVI_raster.tif")


# In[3]:


pixel_based_analysis = input("Conduct Pixel-Based Analisys? (y/n)? ")
if pixel_based_analysis=='n':
    sys.exit(0)


# In[27]:


#COMBINE NDVI AND EVI INDECES TO ONE ARRAY
data_array_combined = (np.append(data_array_ndvi, data_array_evi, axis=0))

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


# In[17]:


score = []


# In[26]:


# neigh_num = 20
# neigh = KNeighborsClassifier(n_neighbors=neigh_num)

# start_train = time.time()
# neigh.fit(X_train, y_train)
# end_train = time.time()

# print(end_train - start_train)

# start_test = time.time()
# result = neigh.score(X_test,y_test)
# end_test = time.time()

# print(end_test - start_test)

# score.append([neigh_num, result])

# print(result)
# print(score)


# In[28]:


# #na allaksw tis parametrous gia na mhn vgainei negative
# linear_svm = LinearSVR()

# start_train = time.time()
# linear_svm.fit(X_train, y_train)
# end_train = time.time()
# print(end_train - start_train)

# start_test = time.time()
# result = linear_svm.score(X_test,y_test)
# end_test = time.time()
# print(result)
# print(end_test - start_test)

