#!/usr/bin/python
# -*- coding:utf-8 -*-
import cv2
import numpy as np
import hashlib

plate_haar = cv2.CascadeClassifier("cascade.xml")
def CarIdentify(img):
    if img.shape[2] == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        print "--------have a gray image!!!!--------"
        gray_img = img
    plates = plate_haar.detectMultiScale(gray_img, 1.08, 2, minSize=(36, 9), maxSize=(36*40, 9*40))

    if len(plates) == 0:
        return False
    else:
        return True


# 给图片赋值一个hash名并保存图片到端口文件夹
def verticalMappingToFolder(image,filename=101):
    name = hashlib.md5(image.data).hexdigest()[:8]
    cv2.imwrite('/home/gmq/HDECLPR/tmp/'+str(filename)+'/'+name+'.png', image)
    # print "Image have saved."


def getvideo(Cam_NO=101):
    videoip = 'rtsp://192.168.253.' + str(Cam_NO)
    cap = cv2.VideoCapture(videoip)
    while (1):
        ret, frame = cap.read()
        if ret:
            if CarIdentify(frame):
            # 如果判断导致的图片质量太差，则改用全部帧保存的方式
            # if 1:
                verticalMappingToFolder(frame, Cam_NO)
        else:
            print "video get false!!!"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    getvideo()