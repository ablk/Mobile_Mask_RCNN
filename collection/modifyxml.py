# -*- coding: utf-8 -*-

from xml.dom import minidom
from xml.dom.minidom import parse
import os

directory="scene_all"
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
        
        continue
    else:
        continue

