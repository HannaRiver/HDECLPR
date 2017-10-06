#!/usr/bin/env python
import cv2
import numpy as np
import hashlib
import os
import shutil
import clpr as clpr
import upload as upload

import urllib
import json
import urllib2


def http_post(PlateINFO, url='http://www.parking.com/'):
    data = urllib.urlencode(PlateINFO)
    data = data.encode('utf-8')
    # print data
    response = urllib.urlopen(url, data)
    the_page = response.read()
    idx_e = the_page.find('err')
    return int(the_page[idx_e+5])

# 加载分类器
plate_haar = cv2.CascadeClassifier("cascade.xml")
car_haar = cv2.CascadeClassifier("car_judgement.xml")

# 判别是不是车，返回bool值
def CarIdentify(img):
    # 把图像转为黑白图像
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_w, img_h = gray_img.shape[0], gray_img.shape[1]
    img_area = img_w * img_h
    # 检测图像总所有的车
    plates = plate_haar.detectMultiScale(gray_img, 1.08, 2, minSize=(36, 9),maxSize=(36*40, 9*40))
    cars = car_haar.detectMultiScale(gray_img)
    cars = []
    
    if len(cars) == 0 and len(plates) == 0:
        return False
    else:
        return True

# 生成一个hash文件名并保存，返回文件名str
def verticalMappingToFolder(image):
    name = hashlib.md5(image.data).hexdigest()[:8]
    # print name

    # cv2.imwrite("./tmp/"+name+".png",image)
    return name

# 获取摄像头信息，并把识别结果上传
def videoget(Cam_NO=175):
    videoip = 'rtsp://192.168.1.' + str(Cam_NO)

    plate2hash = {} # {'沪A99999': ['7a510c61']}
    cap = cv2.VideoCapture(videoip)

    while (1):
        ret, frame = cap.read()
        # cv2.imshow("capture", frame)
        if CarIdentify(frame):
            # 保存有车牌的照片
            image_hash_name = verticalMappingToFolder(frame)
            car_number = clpr.SimpleRecognizePlate(frame) # 返回的只有一个车牌
            if car_number not in plate2hash:
                plate2hash[car_number] = [image_hash_name]
            else:
                plate2hash[car_number] = plate2hash[car_number].append(image_hash_name)
            # 如果一个车牌的次数超过10次，就认为这辆车的车牌为这个
            if len(plate2hash[car_number]) == 10:
                image_url = "/home/gmq/HDECLPR/tmp/"+image_hash_name+".png"
                cv2.imwrite(image_url, frame)
                PlateINFO = clpr.RecognizePlateJson(frame)
                PlateINFO['Cam_NO'] = Cam_NO
                PlateINFO["Image_path"] = image_url
                ans = http_post(PlateINFO)
                if not ans:
                    plate2hash = {}
                    os.remove(image_url)
        # print ret
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    videoget()