import matplotlib.pyplot as plt
import numpy as np
import rasterio
#import os
import sys
import gdal
from osgeo import osr, ogr
import seaborn as sns
import pandas as pd


def array_to_raster(array, old_raster_used_for_projection, save_path):

    width = old_raster_used_for_projection.RasterYSize
    height = old_raster_used_for_projection.RasterXSize
    gt = old_raster_used_for_projection.GetGeoTransform()
    wkt_projection = old_raster_used_for_projection.GetProjectionRef()

    
    if len(array.shape)!=3:
        array = np.expand_dims(array, axis=0)

    no_bands =  array.shape[0]

    
    driver = gdal.GetDriverByName('GTiff')
    DataSet = driver.Create(save_path, height, width, no_bands, gdal.GDT_Float64)
    DataSet.SetGeoTransform(gt)
    DataSet.SetProjection(wkt_projection)
    
    # print(array.shape)
    for i, image in enumerate(array, 1):
        # print(image.shape)
        DataSet.GetRasterBand(i).WriteArray(image)
    DataSet = None
    
    return save_path


def raster_to_vector_polygonize(raster_path, save_path, save_name):
    gdal.UseExceptions()
    src_ds = gdal.Open(raster_path)
    if src_ds is None:
        print('Unable to open %s' % raster_path)
        sys.exit(1)
    srcband = src_ds.GetRasterBand(1) 
    srs = osr.SpatialReference()
    srs.ImportFromWkt(src_ds.GetProjection())

    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(save_path)

    dst_layer = dst_ds.CreateLayer(save_name, srs = srs )
    newField = ogr.FieldDefn('Area', ogr.OFTReal)
    dst_layer.CreateField(newField)
    gdal.Polygonize(srcband, None, dst_layer, 0, [], 
    callback=None )
    
    return save_path


#IMPORT SATELLITE DATA
def read_data_from_1_date(dataset, use_bands, satellite_name):
    #data =[]
    data_list = []
   # files = []

    # number_of_files = len(os.listdir(dataset))
    for i in use_bands:
        temp = rasterio.open(dataset + satellite_name + str(i) + '.tif')
        #data.append(temp)
        data_list.append(temp.read(1).astype('float64'))

                  
#         temp = gdal.Open(dataset + cropped_data + satellite_name + str(i) + '.tif')
#         #data.append(temp)
#         data_list.append(np.array(temp.GetRasterBand(1).ReadAsArray()))  
    number_of_bands = len(data_list)
    print(number_of_bands)
    data_array = np.array(data_list)
    return data_array

#NDVI
def calculate_NDVI(red, nir):

    ndvi = np.where((nir+red)==0., 0, (nir-red)/(nir+red))
    ndvi = ndvi[np.newaxis,:,:]
    return ndvi


#EVI
def calculate_EVI(blue, red, nir):
    evi = np.where((nir + 6*red - 7.5*blue + 1)==0., 0, 2.5*(nir-red)/(nir + 6*red - 7.5*blue + 1))
    evi = evi[np.newaxis,:,:]
    return evi


def calculate_means_of_classes_in_1_band(data, labels):
    dif_labels = np.unique(labels)[1:]
    # print(dif_labels)
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



def spatial_fill_missing_values_in_1_band(data,labels,means,std):
    
    dif_labels = np.unique(labels)[1:]
    no_data = -9999
    no_data_values = data==no_data
    for idx,label in enumerate(dif_labels):
        label_values = labels==label
        values_to_fill = np.logical_and(no_data_values, label_values)
        
        norm_distr_values = np.random.normal(means[idx],std[idx],data.size)
        data = np.where(values_to_fill, norm_distr_values[idx], data)
    return data

def temporal_fill_missing_values(data_list, labels):
    data_list = list(data_list)
    dif_labels = np.unique(labels)[1:]
    no_data = -9999
    
    first_and_last_dates = [data_list[0],data_list[-1]]
    # print(len(first_and_last_dates))
    data = []    
    #fill spatially first and last date 
    for date in first_and_last_dates:
        date_data = []
        for band in date:
            # plt.figure(figsize=(50,20))
            # plt.imshow(band)            
            means,std = calculate_means_of_classes_in_1_band(band, labels)
            band = spatial_fill_missing_values_in_1_band(band, labels, means, std)
            # plt.figure(figsize=(50,20))
            # plt.imshow(band) 
            date_data.append(band)
        data.append(date_data)
    data_list[0] = data[0]
    data_list[-1] = data[-1]
    plt.figure(figsize=(50,20))
    plt.imshow(data_list[0][0]) 
    plt.figure(figsize=(50,20))
    plt.imshow(data_list[0][1]) 
    plt.figure(figsize=(50,20))
    plt.imshow(data_list[-1][0]) 
    plt.figure(figsize=(50,20))
    plt.imshow(data_list[-1][1]) 
    #for every date except first and last

    for date in range(1,len(data_list)-1):
        no_data_values = data_list[date]==no_data
        #for every band in the current date
        date_data = []
        for index,band in enumerate(data_list[date]):

            # plt.figure(figsize=(50,20))
            # plt.imshow(band)
            #for every crop label selected for processing
            for idx,label in enumerate(dif_labels):
                label_values = labels==label
                values_to_fill = np.logical_and(no_data_values[index], label_values)
                
                
                band = np.where(values_to_fill,(data_list[date-1][index] + data_list[date+1][index])/2, band)
                band = np.where(band<-1,no_data,band)


            # plt.figure(figsize=(50,20))
            # plt.imshow(band)



            for idx,label in enumerate(dif_labels):
                label_values = labels==label
                values_to_fill = np.logical_and(no_data_values[index], label_values)
                
                means, std = calculate_means_of_classes_in_1_band(band,labels)
                band = spatial_fill_missing_values_in_1_band(band,labels,means,std)
            date_data.append(band)
        data.insert(-1,date_data)
            # plt.figure(figsize=(50,20))
            # plt.imshow(band)
            
    return np.array(data)

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


def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements