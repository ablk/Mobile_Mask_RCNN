"""
Mobile Mask R-CNN Train & Eval Script
for Training on the COCO Dataset

written by github.com/GustavZ

to use tensorboard run inside model_dir with file "events.out.tfevents.123":
tensorboard --logdir="$(pwd)"
"""

## Import Packages
import os
import sys
import imgaug

## Import Mobile Mask R-CNN
from mmrcnn import model as modellib, utils
import coco
'''
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config));
'''

## Paths
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
WEIGHTS_DIR = os.path.join(ROOT_DIR, "weights")
DEFAULT_WEIGHTS = os.path.join(ROOT_DIR, "mobile_mask_rcnn_coco.h5")

COCO_JSON_TRAIN = os.path.join(ROOT_DIR, 'collection/out_coco_train/annotations.json')
COCO_IMG_DIR_TRAIN = os.path.join(ROOT_DIR, 'collection/out_coco_train')
COCO_JSON_VAL = os.path.join(ROOT_DIR, 'collection/out_coco_val/annotations.json')
COCO_IMG_DIR_VAL = os.path.join(ROOT_DIR, 'collection/out_coco_val')

## Dataset
class_names = ['kinder','kusan','doublemint'] #['person']  # all classes: None
dataset_train = coco.CocoDataset()
dataset_train.load_coco2(COCO_JSON_TRAIN, COCO_IMG_DIR_TRAIN, class_names)
dataset_train.prepare()
print (dataset_train.dataset_size)
dataset_val = coco.CocoDataset()
dataset_val.load_coco2(COCO_JSON_VAL, COCO_IMG_DIR_VAL, class_names)
dataset_val.prepare()


## Model
num_class=4
config = coco.CocoConfig(num_class)
#config.IMAGE_SHAPE=[640,480,3]
config.GPU_COUNT=1
config.BATCH_SIZE=1
config.display()

model = modellib.MaskRCNN(mode="training", model_dir = MODEL_DIR, config=config)
model.keras_model.summary()

## Weights
model_path = model.get_imagenet_weights()
#model_path = model.find_last()[1]
#model_path = DEFAULT_WEIGHTS

print("> Loading weights from {}".format(model_path))
model.load_weights(model_path, by_name=True)

## Training - Config
starting_epoch = model.epoch
epoch = dataset_train.dataset_size // (config.STEPS_PER_EPOCH * config.BATCH_SIZE)+1
print("epoch",epoch)
epochs_warmup = 1* epoch
epochs_heads = 7 * epoch #+ starting_epoch
epochs_stage4 = 7 * epoch #+ starting_epoch
epochs_all = 7 * epoch #+ starting_epoch
epochs_breakOfDawn = 5 * epoch
augmentation = imgaug.augmenters.Fliplr(0.5)
print("> Training Schedule: \
    \nwarmup: {} epochs \
    \nheads: {} epochs \
    \nstage4+: {} epochs \
    \nall layers: {} epochs \
    \ntill the break of Dawn: {} epochs".format(
    epochs_warmup,epochs_heads,epochs_stage4,epochs_all,epochs_breakOfDawn))

## Training - WarmUp Stage
print("> Warm Up all layers")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=epochs_warmup,
            layers='all',
            augmentation=augmentation)

## Training - Stage 1
print("> Training network heads")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=epochs_warmup + epochs_heads,
            layers='heads',
            augmentation=augmentation)

## Training - Stage 2
# Finetune layers  stage 4 and up
print("> Fine tune {} stage 4 and up".format(config.BACKBONE))
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=epochs_warmup + epochs_heads + epochs_stage4,
            layers="4+",
            augmentation=augmentation)

## Training - Stage 3
# Fine tune all layers
print("> Fine tune all layers")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=epochs_warmup + epochs_heads + epochs_stage4 + epochs_all,
            layers='all',
            augmentation=augmentation)

## Training - Stage 3
# Fine tune all layers
print("> Fine tune all layers till the break of Dawn")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 100,
            epochs=epochs_warmup + epochs_heads + epochs_stage4 + epochs_all + epochs_breakOfDawn,
            layers='all',
            augmentation=augmentation)
