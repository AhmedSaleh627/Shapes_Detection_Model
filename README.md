# Shape Detection Model using YOLOv5

[Open in Colab](https://colab.research.google.com/drive/16uytse5hdZFU1QCnMSICPkGqO5xo486f?usp=sharing) -- Here is the Link to the Google Colab Notebook

### Importing Libiraries
```
import os         #For Setting up a dataset directory 
import torch      #To use PyTorch
```
### Installing the required dependencies
```
!git clone https://github.com/ultralytics/yolov5        # clone repo ( YOLOv5 Model )
!pip install -U pycocotools
!pip install -qr yolov5/requirements.txt                # installing the dependencies required for YOLOv5.
!cp yolov5/requirements.txt ./
!pip install roboflow                                    # Installing RoboFlow in which we are going to import our Dataset from.
```
### Importing our dataset from RoboFlow
```
from roboflow import Roboflow
rf=Roboflow(api_key="h2fwpOL5yr87zhweYQwq",model_format="yolov5" , notebook="ultralytics")
os.environ["DATASET_DIRECTORY"]="/content/datasets"
project = rf.workspace("part-o7snh").project("part-syn")          # Our Dataset
dataset = project.version(3).download("yolov5")
```
### Training The Model
```
!python /content/yolov5/train.py --img 640 --batch 8 --epochs 40 --data /content/datasets/part-syn-3/data.yaml --weights yolov5s.pt --cache     # Training the model
```
### Here is the mean average precision of each class to see the accuracy of our model


![map502](https://github.com/AhmedSaleh627/Shapes_Detection_Model/assets/88249795/dedfdc9e-c520-4991-854b-ed50f2ef1af9)



### Here is the results of the training that helps us identify if there is any errors or something wrong with our training, and our main focus here is to ensure the decrease of the losses

![results2](https://github.com/AhmedSaleh627/Shapes_Detection_Model/assets/88249795/c50e4f6a-5e9d-4496-95ec-cbe626b1eef6)


### Also we can see the confusion matrix for more clarification

![confusion_matrix2](https://github.com/AhmedSaleh627/Shapes_Detection_Model/assets/88249795/ee1e5478-8c2d-4e74-83a3-61da2998e653)


### Here is an image from the dataset used
![trainImage](https://github.com/AhmedSaleh627/Shapes_Detection_Model/assets/88249795/1fcfb10f-0117-403f-bdb2-ecf4d8b18932)


### Here is a validation batch prediction

![validation_pred](https://github.com/AhmedSaleh627/Shapes_Detection_Model/assets/88249795/673a3d71-858d-4ca5-b352-e04a6f8224a5)


### Testing the prediction of the model on a seperate test dataset.
```
!python /content/yolov5/detect.py --weights /content/yolov5/runs/train/exp2/weights/best.pt --img 640 --conf 0.25 --source /content/datasets/part-syn-3/test/images   # Testing it on a seperate test
```
###  Here is multiple examples of the output

![no1](https://github.com/AhmedSaleh627/Shapes_Detection_Model/assets/88249795/de225c6a-e5cd-45f0-aae3-c2dcc11133df)

![no2](https://github.com/AhmedSaleh627/Shapes_Detection_Model/assets/88249795/52f2fee5-1098-4002-bb23-f03bcb76e24f)

![no3](https://github.com/AhmedSaleh627/Shapes_Detection_Model/assets/88249795/b459954a-e312-429e-9210-f7ba4904a55c)


### Below is the inference code in which we will take any external image with different distribution and specify it's size to match our dataset ( 640 ) and see the model's prediction on it.
```
!python /content/yolov5/detect.py --weights /content/yolov5/runs/train/exp2/weights/best.pt --img 640 --conf 0.25 --source /content/dc6055f57fc11144481ee3173932158a.jpg    # Testing it on an external image
```

###  Here is the output

![infer](https://github.com/AhmedSaleh627/Shapes_Detection_Model/assets/88249795/0c421db8-c9ed-43bb-a4b5-d1d9039aa51e)


###  Some notes to consider when running the code:
  1-Change the path to the training,testing and validation images in the data.yaml file according to where you saved the dataset.<br/>
  2-Make sure of the path of the model when training ( train.py )
