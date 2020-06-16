#!/usr/bin/env bash

./rename.sh
./move.sh
python modifyxml.py
rm scene_all/scene_000002000002.xml
python voc2coco/xml2coco.py scene_all/ out_coco --labels sis_labels.txt 

