# app-ras

current model and dataset are uploaded on [Google Cloud](https://drive.google.com/drive/folders/1qlWZ4WMYuEwyJQQI8Y9NTPzI9_YT5S5b?usp=sharing)

### DataExtractor2.0.py
The Dataextractor extracts data from a rosbag file and puts it into a dictionary. Before calling the class, we need to make sure that the msgtypes are registered from Carla.

Clone this repository first [Carla_msg](https://github.com/MPC-Berkeley/carla-ros-bridge)
then do
``` python
from pathlib import Path

from rosbags.serde import deserialize_cdr
from rosbags.typesys import get_types_from_msg, register_types

#register carla msgs
control_text = Path('ros-carla-msgs/msg/CarlaEgoVehicleControl.msg').read_text()
status_text = Path('ros-carla-msgs/msg/CarlaEgoVehicleStatus.msg').read_text()
    
add_types = {}
add_types.update(get_types_from_msg(control_text, 'carla_msgs/msg/CarlaEgoVehicleControl'))
add_types.update(get_types_from_msg(status_text, 'carla_msgs/msg/CarlaEgoVehicleStatus'))
register_types(add_types)
```

To create the class, it is easy to call it like the following with a given path to the rosbag datafile.
``` python
pathname = 'path/to/rosbag'
a = DataExtractor(pathname)
```

Configuration or parameters taken by the class are:
* **pathname**: path to rosbag
* **rgb_h, rgb_w:** height and width of the image inside rosbag (or given by carla)
* **depth_h, depth_w:** height and width of depth image
* **manual_control:** bool, whether or not we want to set the command in our model. It is `False` by default, meaning all values would be given as a `0` to indicate going forward only.
* **batch_size:** how many entries do we want in each batch(or in each h5 file)
* **rgb_cropsize, depth_cropsize:** how we want to resize our image
* **save_dir:** where to save the h5 files created by the class
* **save_data**: bool, whether we want to save the h5 file or not, default is `False`

After creating the class, there would be a class member `data` created by the function `initialize()`. `data` is a dictionary with 6 keys:
* rgb_front/image
* depth_front/image
* imu
* speedometer
* command
* labels

after created, we can then run
``` python
a.read_data()
```
to read the data from the file.

#### Real time setup
in case we don't want to read data from a rosbag but directly from carla.
say we want the rgb_front/image.
we could run
``` python
a.read_rgb(rawdata, msgtype)
```
other methods are
``` python
read_depth(rawdata, msgtype)
read_imu(rawdata, msgtype)
read_speed(rawdata, msgtype)
read_status(rawdata, msgtype)
```
the rawdata should be given by the python code (I hope) and the msgtype is `sensor_msgs/msg/Image`. The same works out for the other datas. This function should directly return a numpy (or list) that is finished with cropping or retracting the data that can be input to the model. 
> This function however does not include the data manipulation, so like getting the values standarized are not implemented. This also is a numpy so it needs to be changed to a tensor afterwards.

#### a.flush()
inside the function `read_data()`, I added a function called `flush()` which is not written yet. Normally after the batch_size is reached, I would store the data into a h5 file, but this is only necessary since we are working on different computers and data needs to be transfered from one to another. If all the setup is on one cpu, you could use this `flush()` function to push all the data in the dictionary `a.data` and save it in another tensor or place. It may be good for batches and to keep every batch at a computable size.

#### Save h5 file
h5 file saves the dictionary of `a.data`, but in a slightly different way. Since all the values beside the image are only 1d vectors, we flatten them and store them in the key `others` for reducing the size of the file. 
It is not nessesary if the code is running on only one computer, which then you could save the dictionary as it is with all the same keys or just write the `flush()` function. (I guess you wouldn't need to save them if it's on the same cpu) But in order to save the data as it is, you would need to modify the code in `utils.py`
delete everything **except**
```python
with h5py.File(filename, "w") as file:
        for k, v in dict_.items():
            file.create_dataset(k, data=np.array(v), compression='gzip')
```
Furthermore, code in the `dataloader.py` should also be changed but Fernando has more experience in that part.

### Network.py
although not written by me, I changed some parts to let it print the loss value. Nothing to complicated. 
I saved all the loss in two files though, `train_loss.txt` and also `test_loss.txt`. These two files can then be read in `TestResults.ipynb` to show the pyplot. 
Also, to test the model we trained afterwards with a .pt file, we should do 
``` python
model = Model()
model = torch.load("/path/to/model.pt")
```

### All set
after everything is working, just run 
``` cmd
python network.py
```
and everything should be working.

### Time Dependencies.

current pipeline
``` mermaid
graph TD;
Rawdata-->DataExtraction;
DataExtraction-->Dataloader;
Dataloader-->Model_Output;
```
this time is based on an average of 10 interations

From Rawdata to DataExtraction takes: **0.011558** secs 

From DataExtraction to Dataloader takes: **1.155097** secs 

From Dataloader to Model_output takes: **0.015224** secs 

In total: **1.181880** secs 
