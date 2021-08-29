# IRP-Johnson

Implementation of U-net in Crater automatic detection.

## Datasets
Link to the crated datasets in this project: https://drive.google.com/file/d/1lxiI3inP-xfHm9wuAY9MsMUd_BZ9Omr_/view?usp=sharing .

The file structure:
* mask: This folder contains the original json file created by Labelme
* data_image_1: The crater image with augmentation.
* data_targets_1: The mask image with augmentation.



## Getting Started
The model was coded as python module
### Prerequisites

* Keras ( recommended version : 2.4.3 )
* OpenCV for Python
* Tensorflow ( recommended  version : 2.4.1 )


### Build models and load weights:
```python
from model import UNet
unet = UNet()
unet.load_weights('your_model_weight_pth')
#trained model weights in this project is on './weights/model19_sm_unet_iou_nearest_binary_4.h5'
```


### Using the python module
You can see the example in the expamle.ipynb
You need to complied the model before testing and trainin the model.
If you would like to train the model by yourselves, please complie the model before start the training process. 
Checking the doc string to see the details of the function parameters.
```python
from model import UNet
unet = UNet()
unet.complie()
unet.train(train_img, train_mask, val_img, val_mask)
```
The training img array in this project is [n, 416, 416, 1], and the mask array shape in this project is [n, 416*416, 2].










