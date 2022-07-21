import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as T
#import random
import torchvision.models as models
# from torchsummary import summary
import numpy as np
from dataloader import Train


from PIL import Image
import cv2

class Model(nn.Module):
    def __init__(self):
        ###INPUT MODULES###
        #mobile net takes images of size (224x224x3) as input
        
        super(Model,self).__init__()
        self.mobile_net = models.mobilenet_v2(num_classes=512)
        self.image_module = nn.Sequential(
            nn.Linear(512,1024),
            nn.ReLU(True),
            nn.Linear(1024,512),
            nn.ReLU(True),

            )
        self.imu_module = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(True),
            nn.Linear(256,256),
            nn.ReLU(True),
            )

        self.depth_module = nn.Sequential(#input of size: 200x88)
            nn.Conv2d(1,32,4,2,1,bias=True),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32,32,4,2,1,bias=True),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32,64,4,2,1,bias=True),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64,64,4,2,1,bias=True),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
          #  nn.Conv2d(64,128,4,2,1,bias=False),
          #  nn.Dropout(p=0.2),
          #  nn.BatchNorm2d(128),
          #  nn.ReLU(True),
          #  nn.Conv2d(128,128,4,2,1,bias=False),
          #  nn.Dropout(p=0.2),
          #  nn.BatchNorm2d(128),
          #  nn.ReLU(True),

          #  nn.Conv2d(128,256,4,2,1,bias=False),
          #  nn.Dropout(p=0.2),
          #  nn.BatchNorm2d(256),
          #  nn.ReLU(True),
          #  nn.Conv2d(256,256,4,2,1,bias=False),
          #  nn.Dropout(p=0.2),
          #  nn.BatchNorm2d(256),
          #  nn.ReLU(True),
            
        
            nn.Flatten(),

            #input to linear layer probably incorrect
           # nn.Linear(384 , 512),
           #  nn.Linear(3840,512),
            nn.Linear(3072,512),
            nn.Dropout(p=0.5),
            nn.ReLU(True),

            nn.Linear(512,512),
            nn.Dropout(p=0.5),
            nn.ReLU(True)
            )

        print("SUMMARY DEPTH MODULE")
        # print(summary(self.depth_module,(1,88,200)))
        self.speed_module = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(True),
            nn.Linear(128,128),
            nn.ReLU(True)
            )

        self.dense_layers = nn.Sequential(
            #512depth, 512image, 128speed, 256imu
            nn.Linear(1408,512),
            nn.ReLU(True)
            )

        ###COMMAND BRANCHEs###
        self.straight_net = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(True),
            nn.Linear(256,256),
            nn.ReLU(True),
            nn.Linear(256,3),  #outputlayer is supposed to have no activation function
            nn.Sigmoid()
            )

        self.right_net = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(True),
            nn.Linear(256,256),
            nn.ReLU(True),
            nn.Linear(256,3),  #outputlayer is supposed to have no activation function
            nn.Sigmoid()
            )
        self.left_net = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(True),
            nn.Linear(256,256),
            nn.ReLU(True),
            nn.Linear(256,3),  #outputlayer is supposed to have no activation function
            nn.Sigmoid()
            )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            print("Block:", block)
            if block=='mobile_net':
                pass
            else:
                for m in self._modules[block]:
                    print("M:", m)
                    if isinstance(m, nn.Conv2d):
                        nn.init.normal(m.weight,mean=0, std=0.01)
                    elif isinstance(m, nn.Linear):
                        nn.init.normal(m.weight,mean=0,std=0.01)
                    else:
                        pass
               # normal_init(m)

    def forward(self, input_data):
        image = input_data[0]
        image = self._image_module(image)
        imu   = input_data[2]
        imu   = self._imu_module(imu)
        depth = input_data[1]
        depth = self._depth_module(depth)
        speed = input_data[3]
        speed = self._speed_module(speed)
        command=input_data[4]
        
        concat = torch.cat((image,imu,depth,speed),1)
        concat = self._dense_layers(concat)
        
        ##########################################################
        ##1 here needs to be changed if batch size is changed!!##
        ##########################################################
        output = torch.Tensor()
        for i in range(1):
            if command[i]==1:
                if output.shape==torch.Size([0]):
                    output = self._straight_net(concat)
                else:
                    torch.stack(tensors=(output, self._straight_net(concat)),dim=1)

            elif command[i]==2:
                if output.shape==torch.Size([0]):
                    output = self._right_net(concat)
                else:
                    torch.stack(tensors=(output, self._right_net(concat)),dim=1)

            elif command[i]==0:
                if output.shape==torch.Size([0]):
                    output = self._left_net(concat)
                else:
                    torch.stack(tensors=(output, self._left_net(concat)),dim=1)

        return output

    
    def _image_module(self,x):
        x = self.mobile_net(x)
        return self.image_module(x)
    def _imu_module(self,x):
        return self.imu_module(x)

    def _depth_module(self,x):
        return self.depth_module(x)

    def _speed_module(self,x):
        return self.speed_module(x)

    def _dense_layers(self,x):
        return self.dense_layers(x)

    def _straight_net(self,x):
        return self.straight_net(x)

    def _right_net(self,x):
        return self.right_net(x)

    def _left_net(self,x):
        return self.left_net(x)

def normal_init(m,mean_std):
    if isinstance(m,(nn.Linear,nn.Conv2d)):
        m.weight.dtaa.normal_(mean_std)
        if m.bias.data is not None:
            m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.weight.data.fill_(1)
            if m.bias.data != None:
                m.bias.data.zero_()


class Solver:
    def __init__(self):
        self.MSE_loss = torch.nn.MSELoss()
        self.cuda = True
        self.num_repetitions_per_step = 1
        self.num_train_iterations =  int(200/1) #int(size_dataset/batch_size)
        self.num_test_iterations = int(200/1) #int(size_testdataset/batchsize)
        
        self.loss_steer = []
        self.loss_throttle = []
        self.loss_brake = []

    def read_data(self,dataloader):
        data,label = next(iter(dataloader))
        # print("dataloader", np.asarray(data).shape)
        return data,label

    def train_iteration(self,model, dataloader):
        memory = []
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        model.train()
        if self.cuda==True:
            model = model.cuda()
        for step in range(self.num_repetitions_per_step):
            print("num_repetition inside train_step")
            for i in range(self.num_train_iterations):
                
                batch_data, batch_label  = self.read_data(dataloader)
                batch_label = torch.FloatTensor(batch_label)
                
                if self.cuda==True:
                    for j in range(len(batch_data)):
                        batch_data[j] = batch_data[j].cuda()
                    batch_label = batch_label.cuda()
                
                output = model(batch_data)
                loss = self.MSE_loss((output), batch_label)
                print(f"[Training iter:{i}] loss={loss.item()}")
                memory.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
 #               break
        return model, memory

    def test_iteration(self,model,dataloader):
        model.eval()
        list_mse = []
        for i in range(self.num_test_iterations):
            batch_data, batch_label = self.read_data(dataloader)
            batch_label = torch.FloatTensor(batch_label)
            if self.cuda==True:
                for j in range(len(batch_data)):
                    batch_data[j]  = batch_data[j].cuda()
                batch_label = batch_label.cuda()

            output      = model(batch_data)
            print(f"output of Model: {output.detach()}, Ground Truth: {batch_label}")
            with open("testing.txt", "a") as f:
                f.write(f"{output[0][1]},{batch_label[1]},{output[0][0]},{batch_label[0]},{output[0][2]},{batch_label[2]}\n")
            mse_step   = self.MSE_loss(output,batch_label)
#             print(f"[Testing iter:{i}] loss={mse_step.item()}")
            list_mse.append(mse_step.item())
        avg_mse = np.average(list_mse)
        return avg_mse, list_mse

# ------

# ---------------​

class Test(Dataset):
    def __init__(self):
        super().__init__()
        self.dataset =[]
        transform = T.ToTensor()
        for i in range(150):
            image   = np.random.rand(135,278,3).astype('float32')
            image   = numpy_to_pil(image)
            image   = T.Resize((224,224),
                    interpolation=T.InterpolationMode.BICUBIC)(image)
            image   = transform(image)
           # print("____")
           # print(np.asarray(image).shape)

            depth   = np.random.rand(210,95,1).astype('float32')
            depth   = cv2.resize(depth,dsize=(200,88),
                    interpolation=cv2.INTER_CUBIC)
            depth   = transform(depth)
           # print(np.asarray(depth).shape)

            imu     = np.random.rand(10).astype('float32')
           # print(np.asarray(imu).shape)
            speed   = np.random.rand(1).astype('float32')
           # print(np.asarray(speed).shape)
            command = np.random.randint(0,3, size=(1)).astype('float32')
            label   = np.random.rand(2).astype('float32')
                
            data    = [image,depth,
                torch.Tensor(imu),torch.Tensor(speed),
                torch.Tensor(command)]
            datapoint   = [data,label]
            self.dataset.append(datapoint)

    def __getitem__(self,index):
        data, label = self.dataset[index]
        #print("data",np.asarray(data).shape)
        return data,label

    def __len__(self):
        return len(self.test_dataset)

def numpy_to_pil(image):
    minv = np.amin(image)
    maxv = np.amax(image)
    img  = Image.fromarray((255* (image-minv) / (maxv- minv)).astype(np.uint8))
    return img

if __name__=="__main__":
    print("STARTING")
    # Change train flag to "train" to do image augmentation for rgb images
    train_data  = Train(data_dir='./training_data/', train_eval_flag="no_train")
    # test_data   = Test()
    # Change the data_dir for test
    test_data  = Train(data_dir='./training_data/', train_eval_flag="no_train")

    print("INITIALIZED TRAIN AND TEST SET")
    train_indices = list(range(200)) #as many as train samples
    test_indices = list(range(200)) #as many as test samples
    train_dataloader = DataLoader(train_data,
            sampler=SubsetRandomSampler(train_indices), 
            batch_size=1, drop_last=True)
    data_train = torch.utils.data.DataLoader(
    Train(
        data_dir='.',
        train_eval_flag="train")
    ,
    batch_size=2,
    num_workers=1,
    pin_memory=True,
    shuffle=True
    )
    test_dataloader = DataLoader(test_data,
            sampler=SubsetRandomSampler(test_indices), 
            batch_size=1, drop_last=True)
  #  if True:
   #     stop
    model = Model()
    
    # ***** if you want to use the model we trained before, just uncomment the next line ********
    # model = torch.load("model.pt")
   # print(summary(model,(5,)))
   #  stop
    solver= Solver()
    list_test_loss = []
    list_train_loss = []
    for i in range(2):
        print("##############ITERATION NO {}###############".format(i))
        model, train_loss = solver.train_iteration(model,train_dataloader)
        list_train_loss += train_loss
        torch.save(model, "./model.pt")
        test_loss_avg, test_loss = solver.test_iteration(model, test_dataloader)
        print("Average Lost of ITER {} - {}".format(i, test_loss_avg))
        list_test_loss += test_loss

    # ***********for saving loss to txt file to read*****************
    # with open('train_loss.txt', 'w') as f:
    #     for item in list_train_loss:
    #         f.write("%s\n" % item)
            
    # with open('test_loss.txt', 'w') as f:
    #     for item in list_test_loss:
    #         f.write("%s\n" % item)

