#!/usr/bin/env bash
./rename.sh
./move.sh
rm scene_all/scene_000002000002.xml
rm scene_all/scene_000002000002.jpg
mkdir scene_new
python modifyxml.py

mkdir scene_all/val
mv scene_all/*0.jpg scene_all/val
mv scene_all/*0.xml scene_all/val

mkdir scene_all/train
mv scene_all/*.jpg scene_all/train
mv scene_all/*.xml scene_all/train
mv scene_new/*.jpg scene_all/train
mv scene_new/*.xml scene_all/train

python voc2coco/xml2coco.py scene_all/val out_coco_val --labels sis_labels.txt
python voc2coco/xml2coco.py scene_all/train out_coco_train --labels sis_labels.txt
