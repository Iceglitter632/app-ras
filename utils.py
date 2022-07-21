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
    with h5py.File(filename, "w") as file:
        for k, v in dict_.items():
            file.create_dataset(k, data=np.array(v), compression='gzip')
    return


# In[ ]:




