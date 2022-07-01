# Some basic setup:
# Setup detectron2 logger
# To install detectron RUN setup.sh
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import wget

url="http://images.cocodataset.org/val2017/000000439715.jpg"
# url="https://www.spinny.com/blog/wp-content/uploads/2020/02/Electric-car-charging-1200x800.jpg"
if not os.path.exists("input.jpg"):
    wget.download(url,out="input.jpg")
im = cv2.imread("Images/DSC_6595.JPG")
# cv2_imshow(im)

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

def DrawMask(outputs,cfg):
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("out.jpg",out.get_image()[:, :, ::-1])
    # print(outputs)

def makeTransperent(image,mask,classname,classes):
    assert all([clan in classes for clan in classname]),"{} not Supported".format(classname)
    classIndex=[classes.index(cln) for cln in classname]
    predclass=mask.pred_classes.numpy()
    print(predclass)
    pred_masks=mask.pred_masks.numpy()
    output=np.zeros(image.shape[:2]).astype(np.bool)
    for clas,massk in zip(predclass,pred_masks):
        if clas in classIndex:
            output=np.logical_or(output,massk)
    output=output.astype(image.dtype)[:,:,np.newaxis]*255
    image=np.concatenate((image,output),axis=2)
    cv2.imwrite("Images/out.png",image)



# We can use `Visualizer` to draw the predictions on the image.
classes = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2).metadata.get("thing_classes", None)
print("Available Classes")
print(classes)
makeTransperent(im,outputs["instances"].to("cpu"),["person"],classes)
