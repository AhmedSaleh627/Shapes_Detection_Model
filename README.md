# Shape Detection Model
## Importing Libiraries
```
import os
import torch
```
#### Installing the required dependencies
```
!git clone https://github.com/ultralytics/yolov5  # clone repo
!pip install -U pycocotools
!pip install -qr yolov5/requirements.txt  # install dependencies
!cp yolov5/requirements.txt ./
!pip install roboflow
```
#### Importing our dataset from RoboFlow
```
from roboflow import Roboflow
rf=Roboflow(api_key="h2fwpOL5yr87zhweYQwq",model_format="yolov5" , notebook="ultralytics")
os.environ["DATASET_DIRECTORY"]="/content/datasets"
project = rf.workspace("part-o7snh").project("part-syn")
dataset = project.version(3).download("yolov5")
```
#### Training The Model
```
!python /content/yolov5/train.py --img 640 --batch 8 --epochs 40 --data /content/datasets/part-syn-3/data.yaml --weights yolov5s.pt --cache
```

#### Testing our model
```
!python /content/yolov5/detect.py --weights /content/yolov5/runs/train/exp2/weights/best.pt --img 640 --conf 0.25 --source /content/dc6055f57fc11144481ee3173932158a.jpg
```
