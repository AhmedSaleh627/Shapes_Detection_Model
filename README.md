# Shape Detection Model

[Open in Colab](https://colab.research.google.com/drive/16uytse5hdZFU1QCnMSICPkGqO5xo486f?usp=sharing)

##### Importing Libiraries
```
import os         #For Setting up a dataset directory 
import torch      #To use PyTorch
```
#### Installing the required dependencies
```
!git clone https://github.com/ultralytics/yolov5        # clone repo ( YOLOv5 Model )
!pip install -U pycocotools
!pip install -qr yolov5/requirements.txt                # installing the dependencies required for YOLOv5.
!cp yolov5/requirements.txt ./
!pip install roboflow                                    # Installing RoboFlow in which we are going to import our Dataset from.
```
#### Importing our dataset from RoboFlow
```
from roboflow import Roboflow
rf=Roboflow(api_key="h2fwpOL5yr87zhweYQwq",model_format="yolov5" , notebook="ultralytics")
os.environ["DATASET_DIRECTORY"]="/content/datasets"
project = rf.workspace("part-o7snh").project("part-syn")          # Our Dataset
dataset = project.version(3).download("yolov5")
```
#### Training The Model
```
!python /content/yolov5/train.py --img 640 --batch 8 --epochs 40 --data /content/datasets/part-syn-3/data.yaml --weights yolov5s.pt --cache     # Training the model
```
#### Here is the mean average precision of each class to see the accuracy of our model

![map50](https://github.com/AhmedSaleh627/Eagles/assets/88249795/b04c9970-6723-4764-b1e9-ec396de435e7)


#### Here is the results of the training that helps us identify if there is any errors or something wrong with our training

![results](https://github.com/AhmedSaleh627/Eagles/assets/88249795/87ca6c81-c9a8-4e26-b55a-5945b468565a)




#### Also we can see the confusion matrix for more clarification


#### Here is an image from the dataset used


#### Testing our model
```
!python /content/yolov5/detect.py --weights /content/yolov5/runs/train/exp2/weights/best.pt --img 640 --conf 0.25 --source /content/dc6055f57fc11144481ee3173932158a.jpg    # Testing it on an external image
```
