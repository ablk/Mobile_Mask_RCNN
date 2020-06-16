#!/usr/bin/env bash

cd ./Annotations/users/sis2020/competition_dataset/

for f in *; do
    if [ -d "$f" ]; then
        # Will not run if no directories are available
        echo "$f"
        cd $f
        for ff in *; do mv $ff ${ff/frame-/$f}; done
        ##rename 's/frame-/"$f"/' *.xml
        cd ..
    fi
done

cd ../../../../Images/users/sis2020/competition_dataset/

for f in *; do
    if [ -d "$f" ]; then
        # Will not run if no directories are available
        echo "$f"
        cd $f
        for ff in *; do mv $ff ${ff/frame-/$f}; done
        ##rename 's/frame-/"$f"/' *.xml
        cd ..
    fi
done
