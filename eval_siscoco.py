#!/usr/bin/env python
import rospy

# Import Packages
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import sys
import random
import numpy as np

# Import Mobile Mask R-CNN
from mmrcnn import model as modellib, utils, visualize
from mmrcnn.model import log
import coco
import cv2


# Paths
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_DIR = os.path.join(ROOT_DIR, 'data/coco')
DEFAULT_WEIGHTS = os.path.join(ROOT_DIR, "mobile_mask_rcnn_coco.h5")
NUM_EVALS = 10
COCO_JSON = os.path.join(ROOT_DIR, 'collection/out_coco_val/annotations.json')
COCO_IMG_DIR = os.path.join(ROOT_DIR, 'collection/out_coco_val')

# Load Model
num_class=4
config = coco.CocoConfig(num_class)
config.GPU_COUNT=1
config.BATCH_SIZE=1
config.display()

model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
#model_path = DEFAULT_WEIGHTS
#model_path = model.find_last()[1]
model_path=os.path.join(MODEL_DIR, "512_coco20200619T1629/mask_rcnn_512_coco_0027.h5")

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("> Loading weights from {}".format(model_path))
model.load_weights(model_path, by_name=True)

class_names=['0','kinder','kusan','doublemint']

#original_image=cv2.imread("collection/test/frame0078.jpg")
#original_image=cv2.imread("collection/out_coco_val/JPEGImages/scene_000000000020.jpg")

original_image=cv2.imread("collection/test/testa2.jpg")
original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
results = model.detect([original_image], verbose=1)
r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])


