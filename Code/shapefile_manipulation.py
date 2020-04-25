#!/usr/bin/env python
# coding: utf-8

# In[26]:

import os
from osgeo import gdal
import csv


import rasterstats as rs
import pathlib
os.chdir(str(pathlib.Path(__file__).parent.absolute()))
import functions as fu



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

metrics_list = ["Precision", "Recall", "F1 Score"]
indeces = ["ndvi"]
# In[3]:

txt_name = str(input('Name of the file to write results: '))
file = open(txt_name,"w")
file.write(area[1:] + "\n\n")


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
file.write("Threshold used for parcel detection: " + str(threshold) + "\n\n")
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


    print("open tif")
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
    print("seperate bands and calculate stats from shapefiles")
   #OPEN SEPERATE BAND TIFS AND CALCULATE STATISTICS USING SHAPEFILES
    stats_dicts_for_each_crop_and_date = []
    for shape in shapefiles_name_list:
        print(shape)
        summed_file_path = shapefiles_folder_path + shape
        print(summed_file_path)
        stats_of_each_date = []
        for i in range(len(data_tif)):
            print(i)
            stats_of_each_date.append(rs.zonal_stats(summed_file_path, data_folder_path + separate_bands_folder_path +  str(index) + '_date_' + str(i+1), stats="mean", nodata=-9999))
        stats_dicts_for_each_crop_and_date.append(stats_of_each_date)
    print("save stats and csvs")
    #SAVE THE STATISTICS AS CSVs
    for idx,crop in enumerate(crop_names_list):  
        for i in range(len(stats_dicts_for_each_crop_and_date[idx])):
            toCSV = stats_dicts_for_each_crop_and_date[idx][i]
            keys = toCSV[0].keys()
            with open(csv_path + crop_names_list[idx] + '_date_' + str(i+1) + str('_' + index) + '_stats.csv', 'w') as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(toCSV)
file.close()
