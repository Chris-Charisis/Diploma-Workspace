import numpy as np
import rasterio
import os
import gdal
from osgeo import osr, ogr

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
    files = []

    number_of_files = len(os.listdir(dataset))
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



def fill_missing_values_in_1_band(data,labels,means,std):
    
    dif_labels = np.unique(labels)
    no_data = -9999
    no_data_values = data==no_data
    for idx,label in enumerate(dif_labels):
        label_values = labels==label
        values_to_fill = np.logical_and(no_data_values, label_values)
        
        norm_distr_values = np.random.normal(means[idx],std[idx],data.size)
        data = np.where(values_to_fill, norm_distr_values[idx], data)
    return data


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