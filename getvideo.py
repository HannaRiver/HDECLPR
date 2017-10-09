#!/usr/bin/python
# -*- coding:utf-8 -*-
import cv2
import numpy as np
import hashlib


# 给图片赋值一个hash名并保存图片到端口文件夹
def verticalMappingToFolder(image,filename=101):
    name = hashlib.md5(image.data).hexdigest()[:8]
    cv2.imwrite("/home/gmq/HDECLPR/tmp/"+str(filename)+"/"+name+".png", image)
    print "Image have saved."


def getvideo(Cam_NO=101):
    videoip = 'rtsp://192.168.253.' + str(Cam_NO)
    cap = cv2.VideoCapture(videoip)
    while (1):
        ret, frame = cap.read()
        if ret:
            verticalMappingToFolder(frame, Cam_NO)
        else:
            print "video get false!!!"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    getvideo()