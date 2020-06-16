#!/usr/bin/env bash

f=scene_all
mkdir $f
find ./Annotations/users/sis2020/competition_dataset/ -type f -print0 | xargs -0 cp -t ./scene_all
find ./Images/users/sis2020/competition_dataset/ -type f -print0 | xargs -0 cp -t ./scene_all

