#!/usr/bin/env python
# coding: utf-8

# In[7]:


import sys
import numpy as np

from skimage.transform import resize
from utils import rgba2gray, rgba2rgb, storetable

# rosbag dependencies
from pathlib import Path

from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from rosbags.typesys import get_types_from_msg, register_types

class DataExtractor():
    def __init__(self, pathname, rgb_h=600, rgb_w=800, manual_control = False,                  depth_h=70, depth_w=400, batch_size=200,                  rgb_cropsize=(224,224,3), depth_cropsize=(70,200),                  save_dir = "./train_data", save_data = False):
        self.pathname = pathname
        self.rgb_h = rgb_h
        self.rgb_w = rgb_w
        self.depth_h = depth_h
        self.depth_w = depth_w
        self.manual_control = manual_control
        self.batch_size = batch_size
        self.rgb_cropsize = rgb_cropsize
        self.depth_cropsize = depth_cropsize
        self.save_dir = save_dir
        self.save_data = save_data
        
        self.data = {}
        
        self.get_topics()
        self.initialize()
        
    def get_topics(self):
        self.topics = []
        self.msg_count = sys.maxsize
        with Reader(self.pathname) as reader:
            for connection in reader.connections:
                self.topics.append(connection.topic)
                if connection.msgcount < self.msg_count:
                    self.msg_count = connection.msgcount
        self.num_topics = len(self.topics)

    def initialize(self):
        self.data['labels'] = np.empty((0, 3))
        self.data['command'] = []
        for t in self.topics:
            k = t.split('ego_vehicle/')[-1]
            if k == 'rgb_front/image':
                self.data[k] = np.empty((0, self.rgb_cropsize[0], self.rgb_cropsize[1], 3))
            elif k == 'depth_front/image':
                self.data[k] = np.empty((0, self.depth_cropsize[0], self.depth_cropsize[1]))
            elif k == 'imu':
                self.data[k] = np.empty((0, 10))
            elif k == 'speedometer':
                self.data[k] = []
    
    def read_data(self):
        with Reader(self.pathname) as reader:
            count = 0
            for connection, timestamp, rawdata in reader.messages():
                k = connection.topic.split('ego_vehicle/')[-1]
                if k == 'rgb_front/image':
                    msg = self.read_rgb(rawdata, connection.msgtype)
                    msg = np.expand_dims(msg, axis=0)
                    self.data[k] = np.append(self.data[k], msg, axis=0)
                elif k == 'depth_front/image':
                    msg = self.read_depth(rawdata, connection.msgtype)
                    msg = np.expand_dims(msg, axis=0)
                    self.data[k] = np.append(self.data[k], msg, axis=0)
                elif k == 'speedometer':
                    msg = self.read_speed(rawdata, connection.msgtype)
                    self.data[k].append(msg)
                elif k == 'imu':
                    msg = self.read_imu(rawdata, connection.msgtype)
                    msg = np.expand_dims(msg, axis=0)
                    self.data[k] = np.append(self.data[k], msg, axis=0)
                elif k == 'vehicle_status':
                    labels, cmd = self.read_status(rawdata, connection.msgtype)
                    labels = np.expand_dims(labels, axis=0)
                    self.data['labels'] = np.append(self.data['labels'], labels, axis=0)
                    self.data['command'].append(cmd)   

                count+=1
                
                if (count%(self.batch_size*self.num_topics)==0 or count==(self.msg_count*self.num_topics)):
#                     flush()
                    if self.save_data:
                        filname = f'{self.save_dir}/data_{count//self.num_topics}.h5'
                        storetable(filname, self.data)
                        self.initialize()     
                
    
    def read_rgb(self, request, msgtype):
        msg = deserialize_cdr(request, msgtype)
        img = np.reshape(msg.data, (self.rgb_h, self.rgb_w, 4))
        img = rgba2rgb(img)
        img = resize(img, self.rgb_cropsize)
        return img
        
    def read_depth(self, request, msgtype):
        msg = deserialize_cdr(request, msgtype)
        img = np.reshape(msg.data, (self.depth_h, self.depth_w, 4))
        img = rgba2gray(img)
        img = resize(img, self.depth_cropsize)
        return img
        
    def read_speed(self, request, msgtype):
        msg = deserialize_cdr(request, msgtype)
        return msg.data
    
    def read_imu(self, request, msgtype):
        msg = deserialize_cdr(request, msgtype)
        cur = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])
        return cur
    
    def read_status(self, request, msgtype):
        msg = deserialize_cdr(request, msgtype)
        cur = np.array([
            msg.control.throttle,
            msg.control.steer, 
            msg.control.brake
        ])
        # going straight everytime
        if not self.manual_control:
            cmd = 0
#         else:
#             cmd = smth
        return cur, cmd

#     def flush(self):
        # DO SOMETHING
        # Can call this function after n-entries of data
        # Can make batch_sizes

    def get_rgb(self):
        return self.data['rgb_front/image']
    def get_depth(self):
        return self.data['depth_front/image']
    def get_imu(self):
        return self.data['imu']
    def get_speedometer(self):
        return self.data['speedometer']
    def get_cmd(self):
        return self.data['command']
    def get_label(self):
        return self.data['label']


# In[39]:

if __name__ == '__main__':
    
    #register carla msgs
    control_text = Path('ros-carla-msgs/msg/CarlaEgoVehicleControl.msg').read_text()
    status_text = Path('ros-carla-msgs/msg/CarlaEgoVehicleStatus.msg').read_text()
    
    add_types = {}
    add_types.update(get_types_from_msg(control_text, 'carla_msgs/msg/CarlaEgoVehicleControl'))
    add_types.update(get_types_from_msg(status_text, 'carla_msgs/msg/CarlaEgoVehicleStatus'))
    register_types(add_types)
    
    # Call DataExtractor
    a = DataExtractor('carla_withoutstop_5data/')
    a.read_data()







