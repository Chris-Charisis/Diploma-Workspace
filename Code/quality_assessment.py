#!/usr/bin/env python
# coding: utf-8

# In[54]:


import numpy as np
def apply_mask(data,quality_band):
    # 2720 - CLEAR
    # 2752 - CLOUD CONFIDENCE MEDIUM
    # 2800 - CLOUD
    # 2976 - CLOUD SHADOW HIGH
    # 3008 - CIRRUS CONFIDENCE LOW
    data_values = np.array([2720, 2724, 2728, 2732])
    num_of_bands = np.size(data, axis=0)
    print(data.shape)
    print(quality_band.shape)
    
    mask = np.where(np.isin(quality_band,data_values), 1, quality_band) 
    #negative value (-1) because data contain elements with 0 value
    mask = np.where(mask!=1, -1, mask) 

    for i in range(num_of_bands):
        data[i] = np.multiply(data[i], mask)
    
    return np.where(data<0, -9999, data)

