#!/usr/bin/python
# -*- coding:utf-8 -*-
import cv2
import numpy as np
import os
import shutil
import time
import clpr as clpr 

import requests

import sys
reload(sys)
sys.setdefaultencoding("utf-8")


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


def http_post(PlateINFO, url='http://www.parking.com/'):
    r = requests.post(url, data=PlateINFO)
    print r.content
    return r.ok

def processvideo(Cam_NO=101):
    pathtitle = '/home/gmq/HDECLPR/tmp/'
    filepath = pathtitle + str(Cam_NO) + '/'
    plate2hash = {}
    while len(os.listdir(filepath)) >= 0:
        if len(os.listdir(filepath))== 0:
            # print "video get too slow! I should wait a monent."
            time.sleep(1)
        # print "old path", os.listdir(filepath)
        for filename in os.listdir(filepath):
            frame = cv2.imread(filepath+filename)
            # 如果getvideo采用不旁边都保存帧，则这里需要识别
            # if CarIdentify(frame):
            if True:
                car_number = clpr.SimpleRecognizePlate(frame)
                if car_number == '':
                    print "maybe a car, but can not be recognized."
                    os.remove(filepath+filename)
                    continue
                print "have a car, number is:", car_number
                print "can be recognized image name is:", filename
                # shutil.move(filepath+filename, pathtitle+str(Cam_NO)+'r')
                if car_number not in plate2hash:
                    plate2hash[car_number] = [filename]
                else:
                    plate2hash[car_number].append(filename)
                if len(plate2hash[car_number]) >= 3:
                    shutil.move(filepath+filename, pathtitle+str(Cam_NO)+'r')
                    # os.remove(filepath+filename)
                    PlateINFO = {}
                    PlateINFO['Plate_Number'] = car_number
                    PlateINFO['Cam_NO'] = Cam_NO
                    # PlateINFO['Image_Path'] = filepath + filename
                    print "upload INFO:", PlateINFO
                    ans = http_post(PlateINFO)
                    print "web return ans:", ans
                    if ans:
                        plate2hash = {}
                        # 上传成功就清空文件夹
                        for lastfile in os.listdir(filepath):
                            # print "new path", os.listdir(filepath)
                            os.remove(filepath+lastfile)
                        break

                else:
                    os.remove(filepath+filename)
                    pass
            else:
                os.remove(filepath+filename)
        if 0xFF == ord('q'):
            break

if __name__ == '__main__':
    processvideo()
