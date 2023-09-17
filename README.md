# Shape Detection Model

[Open in Colab](https://colab.research.google.com/drive/16uytse5hdZFU1QCnMSICPkGqO5xo486f?usp=sharing)

###### Importing Libiraries
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

#### Testing our model
```
!python /content/yolov5/detect.py --weights /content/yolov5/runs/train/exp2/weights/best.pt --img 640 --conf 0.25 --source /content/dc6055f57fc11144481ee3173932158a.jpg    # Testing it on an external image
```
