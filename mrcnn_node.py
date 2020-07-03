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

import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from sensor_msgs.msg import Image,CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import tensorflow as tf
import time
import math
import message_filters
global graph
graph = tf.get_default_graph()

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
model_path=os.path.join(MODEL_DIR, "512_coco20200619T1629/mask_rcnn_512_coco_0006.h5")

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"

#class_names = ['kinder','kusan','doublemint']
"""
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
"""

class image_converter:

    def __init__(self):

        self.image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        self.depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
        self.cam_info_sub=rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.cam_info_cb)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], 10, 0.5)
        self.ts.registerCallback(self.rosRGBDCallBack)
        
        self.bridge = CvBridge()
        self.fx=None
        self.fy=None
        self.cx=None
        self.cy=None
        
        self.cloud1_pub = rospy.Publisher('mrcnn_pc1', PointCloud, queue_size=10)
        self.cloud2_pub = rospy.Publisher('mrcnn_pc2', PointCloud, queue_size=10)
        self.cloud3_pub = rospy.Publisher('mrcnn_pc3', PointCloud, queue_size=10)
        
        
        
        self.model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
        print("> Loading weights from {}".format(model_path))
        self.model.load_weights(model_path, by_name=True)
        print("done!")
        
        
        
        
    def rosRGBDCallBack(self,rgb_data, depth_data):
        if self.fx is None:
            return


    
        start_time = time.time()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_data, "passthrough")
            cv_depthimage = self.bridge.imgmsg_to_cv2(depth_data, "passthrough")
            cv_depthimage2 = np.array(cv_depthimage, dtype=np.float32)
        except CvBridgeError as e:
            print(e)
        
        with graph.as_default():
            results = self.model.detect([cv_image], verbose=0)
            
            h,w,c = results[0]['masks'].shape
            print("detect:",c)
            for i in range(c):
                #print(i,":",results[0]['class_ids'][i])
                #print(results[0]['masks'][:,:,i].shape)
                
                pc=PointCloud()
                pc.header=rgb_data.header
                pc.header.frame_id="camera_link"
                mask=results[0]['masks'][:,:,i]
                
                tt=0
                for j in range(w):
                    for k in range(h):
                        if mask[k][j] == True:
                            tt=tt+1
                            zc=cv_depthimage2[k][j]
                            if not math.isnan(zc):
                                zc=zc/1000.0
                                pt=Point32()
                                pt.x,pt.y,pt.z=self.getXYZ(k,j,zc)
                                pc.points.append(pt)
                print(results[0]['class_ids'][i],len(pc.points),tt)
                
                if(len(pc.points)>100):
                    if results[0]['class_ids'][i]==1 :
                        #print ("class1")
                        self.cloud1_pub.publish(pc)
                    elif results[0]['class_ids'][i]==2 :
                        #print ("class2")
                        self.cloud2_pub.publish(pc)
                    elif results[0]['class_ids'][i]==3 :
                        #print ("class3")
                        self.cloud3_pub.publish(pc)

        print("--- %s seconds ---" % (time.time() - start_time))

    def getXYZ(self,xp, yp, zc):
        #### Definition:
        # cx, cy : image center(pixel)
        # fx, fy : focal length
        # xp, yp: index of the depth image
        # zc: depth
        inv_fx = 1.0/self.fx
        inv_fy = 1.0/self.fy
        x = (xp-self.cx) *  zc * inv_fx
        y = (yp-self.cy) *  zc * inv_fy
        z = zc
        return (x,y,z)


    def cam_info_cb(self,msg):
        if self.fx is not None:
            return
    
        self.fx = msg.P[0]
        self.fy = msg.P[5]
        self.cx = msg.P[2]
        self.cy = msg.P[6]
        print("cam_info:",self.fx,",",self.fy,",",self.cx,",",self.cy)
        

def main(args):
    ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)