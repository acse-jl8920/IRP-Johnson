# IRP-Johnson

Implementation of U-net in Crater automatic detection.

## Datasets
Link to the crated datasets in this project: https://drive.google.com/file/d/1lxiI3inP-xfHm9wuAY9MsMUd_BZ9Omr_/view?usp=sharing
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


## Build models and load weights:
```python
from model import UNet
unet = UNet()
unet.load_weights('your_model_weight_pth')
#trained model weights in this project is on './weights/model19_sm_unet_iou_nearest_binary_4.h5'
```


## Using the python module
You can see the example in the expamle.ipynb












