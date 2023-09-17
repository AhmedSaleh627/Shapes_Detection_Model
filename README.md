# Shape Detection Model

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

![map502](https://github.com/AhmedSaleh627/Eagles/assets/88249795/179e63e2-1282-45c4-90a7-4ac58e8b6d0f)


### Here is the results of the training that helps us identify if there is any errors or something wrong with our training, and our main focus here is to ensure the decrease of the losses

![results2](https://github.com/AhmedSaleh627/Eagles/assets/88249795/31fc7e49-ecc3-4975-a11d-92bafb59f18f)


### Also we can see the confusion matrix for more clarification

![confusion_matrix2](https://github.com/AhmedSaleh627/Eagles/assets/88249795/acdae02d-7c1a-471d-b3e1-abf1e613b70f)


### Here is an image from the dataset used
![trainImage](https://github.com/AhmedSaleh627/Eagles/assets/88249795/ec1b7c33-11f8-42c0-97b6-d363b2ee4d76)

### Here is a validation batch prediction

![validation_pred](https://github.com/AhmedSaleh627/Eagles/assets/88249795/fa90fc36-15d5-4f67-88ae-5fadc2300d4f)

### Testing the prediction of the model on our test dataset.
```
!python /content/yolov5/detect.py --weights /content/yolov5/runs/train/exp2/weights/best.pt --img 640 --conf 0.25 --source /content/datasets/part-syn-3/test/images   # Testing it on a seperate test
```
###  Here is multiple examples of the output

![no1](https://github.com/AhmedSaleh627/Eagles/assets/88249795/8a310f9e-3384-4a64-bd52-76bd18ee2748)
![no2](https://github.com/AhmedSaleh627/Eagles/assets/88249795/8b1ac462-9636-4320-ad2f-8f717052c102)
![no3](https://github.com/AhmedSaleh627/Eagles/assets/88249795/393aabda-4386-48a1-9334-c5a8e0c80b60)

### Testing the prediction of the model on an external image from different distribution.
```
!python /content/yolov5/detect.py --weights /content/yolov5/runs/train/exp2/weights/best.pt --img 640 --conf 0.25 --source /content/dc6055f57fc11144481ee3173932158a.jpg    # Testing it on an external image
```

###  Here is the output

![infer](https://github.com/AhmedSaleh627/Eagles/assets/88249795/254b1e0a-a7ad-4e1f-896b-ba5fc9ab24ff)

###  Some notes to consider when running the code:
  1-Change the path to the training,testing and validation images in the data.yaml file according to where you saved the dataset.<br/>
  2-Make sure of the path of the model when training ( train.py )
