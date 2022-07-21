from configparser import Interpolation
from tracemalloc import start
# import lmdb
import pandas as pd
# import pyarrow as pa
import numpy as np
import time
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as T
import torch
from imgaug import augmenters as iaa
from helper import RandomTransWrapper
import h5py
import glob

class Train(Dataset):
    def __init__(self, data_dir, train_eval_flag="train", sequence_len=200):
        self.data_dir = data_dir
        self.data_list = glob.glob(data_dir+'*.h5')
        self.data_list.sort()
        self.sequnece_len = sequence_len
        self.train_eval_flag = train_eval_flag

        self.build_transform()


    def build_transform(self):
        if self.train_eval_flag == "train":
            self.transform = T.Compose([
                T.RandomOrder([
                    RandomTransWrapper(
                        seq=iaa.GaussianBlur(
                            (0, 1.5)),
                        p=0.09),
                    RandomTransWrapper(
                        seq=iaa.AdditiveGaussianNoise(
                            loc=0,
                            scale=(0.0, 0.05),
                            per_channel=0.5),
                        p=0.09),
                    RandomTransWrapper(
                        seq=iaa.Dropout(
                            (0.0, 0.10),
                            per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.CoarseDropout(
                            (0.0, 0.10),
                            size_percent=(0.08, 0.2),
                            per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.Add(
                            (-20, 20),
                            per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.Multiply(
                            (0.9, 1.1),
                            per_channel=0.2),
                        p=0.4),
                    RandomTransWrapper(
                        seq=iaa.ContrastNormalization(
                            (0.8, 1.2),
                            per_channel=0.5),
                        p=0.09),
                ]),
                T.ToTensor()])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
            ])

    def __len__(self):
        return self.sequnece_len * len(self.data_list)

    def __getitem__(self, idx):
        data_idx = idx // self.sequnece_len
        file_idx = idx % self.sequnece_len
        file_name = self.data_list[data_idx]
        convert_tensor = T.ToTensor()
        with h5py.File(file_name, 'r') as h5_file:
            img_rgb = np.array(h5_file['rgb_front/image'])[file_idx].astype(np.uint8)
            img_rgb = self.transform(img_rgb)

            img_depth = np.array(h5_file['depth_front/image'])[file_idx].astype(np.uint8)
            img_depth = convert_tensor(img_depth)

            others = np.array(h5_file['others'])[file_idx].astype(np.float32)

            imu = others[:10]

            speed = others[10]

            command = others[11]

            # Acceleration, Brake, Steer
            label = [others[12],  others[14], others[13]]


            # speed = [np.array(h5_file['speedometer'])[file_idx]]
            #
            # status = np.array(h5_file['vehicle_status'])[file_idx]
            #
            # command = np.array(h5_file['command'])[file_idx]
            # [- brake + acc, steer]
            # label = [-status[0] + status[1], status[2]]

            data    = [
                img_rgb,
                img_depth,
                torch.Tensor(imu),
                torch.Tensor([speed]),
                # Adding 0 for now as the command
                torch.Tensor([command])
                # torch.Tensor([0])
                       ]
        return data, label

if __name__ == '__main__':

        # data_train = torch.utils.data.DataLoader(
        data_train = Train(
            data_dir='./training_data/',
            train_eval_flag="train")
            # ,
        # batch_size=2,
        # num_workers=1,
        # pin_memory=True,
        # shuffle=True
            # )

        data, label = data_train.__getitem__(50)
        print(data.shape)