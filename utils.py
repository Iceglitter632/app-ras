#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import h5py

def rgba2rgb(rgba, background=(255,255,255)):
    row, col, ch = rgba.shape
        
    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8')

def rgba2gray(rgba):
    gray = np.mean(rgba[...,:3], -1)
    return gray

def storetable(filename, dict_):

    t1 = dict_['imu']
    t2 = dict_['speedometer']
    t2 = np.array(t2)
    t2 = np.expand_dims(t2, axis=1)
    t3 = dict_['command']
    t3 = np.array(t3)
    t3 = np.expand_dims(t3, axis=1)
    t4 = dict_['labels']
    
    arr = np.concatenate([t1, t2, t3, t4], axis=1)
    
    del dict_['imu']
    del dict_['speedometer']
    del dict_['command']
    del dict_['labels']
    
    dict_['others'] = arr
    
    with h5py.File(filename, "w") as file:
        for k, v in dict_.items():
            file.create_dataset(k, data=np.array(v), compression='gzip')
    return


# In[ ]:




