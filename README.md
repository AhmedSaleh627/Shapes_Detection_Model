# Shape Detection Model using YOLOv5

[Open in Colab](https://colab.research.google.com/drive/16uytse5hdZFU1QCnMSICPkGqO5xo486f?usp=sharing) -- Here is the Link to the Google Colab Notebook where you can find the Python code

YOLOv5 is a popular object detection model known for its speed and accuracy. It is well-suited for real-time applications and can detect various objects, including shapes, in images or videos. YOLOv5 achieves this by dividing the input image into a grid and predicting bounding boxes and class probabilities for each grid cell. This makes it a suitable choice for shape detection tasks where speed and accuracy are important factors.



### Importing Libiraries
```
import os                 # This library is used for setting up the dataset directory.
import torch               # It is used to use the PyTorch library.
```
### Installing the required dependencies
```
!git clone https://github.com/ultralytics/yolov5  # This command clones the YOLOv5 repository from GitHub, which contains the necessary code for using the YOLOv5 model.
!pip install -U pycocotools                       # This command installs the pycocotools library, which is required for working with COCO dataset annotations.
!pip install -qr yolov5/requirements.txt  # install dependencies
!cp yolov5/requirements.txt ./
!pip install roboflow                                      #This command installs the RoboFlow library, which is used to import the dataset.
```
### Importing our dataset from RoboFlow
```
from roboflow import Roboflow
rf=Roboflow(api_key="h2fwpOL5yr87zhweYQwq",model_format="yolov5" , notebook="ultralytics") #This line initializes RoboFlow with the provided API key, model format (yolov5), and notebook name.
os.environ["DATASET_DIRECTORY"]="/content/datasets"          # This line sets the dataset directory to "/content/datasets".
project = rf.workspace("part-o7snh").project("part-syn")     # This line specifies the RoboFlow workspace and project to access the desired dataset.
dataset = project.version(3).download("yolov5")              #This line downloads the dataset from the specified project and version and saves it in the "yolov5" directory.
```
### Training The Model
```
#This command trains the YOLOv5 model using the downloaded dataset. It specifies various training parameters such as image size, batch size, number of epochs, data configuration file, initial weights, and cache.
!python /content/yolov5/train.py --img 640 --batch 8 --epochs 40 --data /content/datasets/part-syn-3/data.yaml --weights yolov5s.pt --cache
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
#This command tests the trained model on a separate test dataset. It specifies the path to the trained weights, image size, confidence threshold, and the source directory containing the test images.
!python /content/yolov5/detect.py --weights /content/yolov5/runs/train/exp2/weights/best.pt --img 640 --conf 0.25 --source /content/datasets/part-syn-3/test/images

```
###  Here is multiple examples of the output

![no1](https://github.com/AhmedSaleh627/Shapes_Detection_Model/assets/88249795/de225c6a-e5cd-45f0-aae3-c2dcc11133df)

![no2](https://github.com/AhmedSaleh627/Shapes_Detection_Model/assets/88249795/52f2fee5-1098-4002-bb23-f03bcb76e24f)

![no3](https://github.com/AhmedSaleh627/Shapes_Detection_Model/assets/88249795/b459954a-e312-429e-9210-f7ba4904a55c)


### Below is the inference code in which we will take any external image with different distribution and specify it's size to match our dataset ( 640 ) and see the model's prediction on it.
```
#This command tests the trained model on an external image. It specifies the path to the trained weights, image size, confidence threshold, and the path to the specific image file.
!python /content/yolov5/detect.py --weights /content/yolov5/runs/train/exp2/weights/best.pt --img 640 --conf 0.25 --source /content/2d-shapes-for-kids-printable-sheet-preschool-vector-37938450.jpg

```

###  Here is the output

![infer](https://github.com/AhmedSaleh627/Shapes_Detection_Model/assets/88249795/0c421db8-c9ed-43bb-a4b5-d1d9039aa51e)


###  Some notes to consider when running the code:
  1-Change the path to the training,testing and validation images in the data.yaml file according to where you saved the dataset.<br/>
  2-Make sure of the path of the model when training and predicting ( train.py , detect.py )
