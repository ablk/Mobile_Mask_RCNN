# -*- coding: utf-8 -*-

from xml.dom import minidom
from xml.dom.minidom import parse
import os
import cv2
import numpy as np

import random
directory="scene_all"
d_new="scene_new"
for filename in os.listdir(directory):
    if filename.endswith(".xml"):
        print(os.path.join(directory, filename))
        #imgname=filename.split('.')[0]+".jpg"
        #print(imgname)
        xmldoc = minidom.parse(os.path.join(directory, filename))
        docelem = xmldoc.documentElement
        imgnm=docelem.getElementsByTagName("filename")
        imgdir=docelem.getElementsByTagName("folder")
        #print(imgnm[0].firstChild.data)
        #print(imgdir[0].firstChild.data)

        imgnm[0].firstChild.data=filename.split('.')[0]+".jpg"
        imgdir[0].firstChild.data="scene_all"
        #print(imgnm[0].firstChild.data)

        usr=docelem.getElementsByTagName("username")
        for i in range(len(usr)):
            usr[i].firstChild.data="anonymous"
            #print(usr[i].firstChild.data)
        
        file_handle = open(os.path.join(directory, filename),"wb")
        xmldoc.writexml(file_handle)
        file_handle.close()


        imgname=filename.split('.')[0]+".jpg"
        imgname=os.path.join(directory, imgname)
        print(imgname)
        img = cv2.imread(imgname)


        mask = np.zeros(img.shape[:2], np.uint8)

        poly=docelem.getElementsByTagName("polygon")
	for p in poly:
            pts=[]
            xdata=p.getElementsByTagName("x")
            ydata=p.getElementsByTagName("y")
            for i in range(len(xdata)):
                u = int(xdata[i].firstChild.data)
                v = int(ydata[i].firstChild.data)
                pts.append([u,v])
            
            ptsarray=np.array(pts)    
            #print ptsarray 
            cv2.drawContours(mask, [ptsarray], -1, (255, 255, 255), -1, cv2.LINE_AA)




        objs = cv2.bitwise_and(img, img, mask=mask)

        nmask = cv2.bitwise_not(mask)
        
        b1 = nmask * random.randint(0,255)
        g1 = nmask * random.randint(0,255)
        r1 = nmask * random.randint(0,255)
        bg1 = cv2.merge([b1,g1,r1])
	new1 = bg1 + objs

        mean = 0
        var = 100
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, img.shape[:2])

        new2 = np.zeros(img.shape, np.float32)

        new2[:, :, 0] = img[:, :, 0] + gaussian
        new2[:, :, 1] = img[:, :, 1] + gaussian
        new2[:, :, 2] = img[:, :, 2] + gaussian

        cv2.normalize(new2, new2, 0, 255, cv2.NORM_MINMAX, dtype=-1)
        new2 = new2.astype(np.uint8)

        bg3_path=os.path.join("background", random.choice(os.listdir("background")))
        print(bg3_path)
        bg3 = cv2.imread(bg3_path,cv2.IMREAD_COLOR)
        bg3 = cv2.resize(src=bg3,dsize=(640,480))
        bg3 = cv2.bitwise_and(bg3,bg3, mask=nmask)
        new3 = bg3 + objs


        bg4 = cv2.bitwise_not(img,mask=nmask)
        new4 = bg4+objs



        newnm=filename.split('.')[0]+"a"+".jpg"
        cv2.imwrite(os.path.join(d_new, newnm), new1)
        file_handle = open(os.path.join(d_new, filename.split('.')[0]+"a"+".xml"),"wb")
        imgnm[0].firstChild.data=newnm
        xmldoc.writexml(file_handle)
        file_handle.close()

        newnm=filename.split('.')[0]+"b"+".jpg"
        cv2.imwrite(os.path.join(d_new, newnm), new2)
        file_handle = open(os.path.join(d_new, filename.split('.')[0]+"b"+".xml"),"wb")
        imgnm[0].firstChild.data=newnm
        xmldoc.writexml(file_handle)
        file_handle.close()


        newnm=filename.split('.')[0]+"c"+".jpg"
        cv2.imwrite(os.path.join(d_new, newnm), new3)
        file_handle = open(os.path.join(d_new, filename.split('.')[0]+"c"+".xml"),"wb")
        imgnm[0].firstChild.data=newnm
        xmldoc.writexml(file_handle)
        file_handle.close()


        newnm=filename.split('.')[0]+"d"+".jpg"
        cv2.imwrite(os.path.join(d_new, newnm), new4)
        file_handle = open(os.path.join(d_new, filename.split('.')[0]+"d"+".xml"),"wb")
        imgnm[0].firstChild.data=newnm
        xmldoc.writexml(file_handle)
        file_handle.close()
        
        continue
    else:
        continue

