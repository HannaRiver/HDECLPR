#!/usr/bin/env python3
import clpr as clpr
import car_identify as caride
import cv2
import imageio

import urllib
import json
import urllib.request

import imageio
import logging


def http_post(url='http://www.parking.com/'):
    values = str(PlateINFO)
    jdata = json.dumps(values)
    #jdata = jdata.encode('utf-8')
    jdata = "plateOut=" + jdata
    # jdata = "plateOut=" + values
    jdata = jdata.encode('utf-8')
    # print("Type:", type(jdata))
    # print(jdata)
    req = urllib.request.Request(url, jdata) # 生成页面请求的完整数据
    response = urllib.request.urlopen(req) # 发送页面请求
    return response.read()

# 上传表单格式
def http_post1(url='http://www.parking.com/'):
    data = urllib.parse.urlencode(PlateINFO)
    data = data.encode('utf-8')
    # print(data)
    # req = urllib.request.Request(url, data)
    # response = urllib.request.urlopen(req)
    response = urllib.request.urlopen(url, data)
    the_page = response.read()
    # print(the_page)

if __name__ == '__main__':
    img_path = 'JingH99999.jpg'
    image = cv2.imread(img_path)
    # CarINFO = {}
    # plateOut = {}
    CarINFO = caride.CarIdentify(image)
    #print(CarINFO)
    if CarINFO['isCar']:
        PlateINFOs = clpr.clpr(img_path)
        PlateINFO = PlateINFOs[0]
        PlateINFO['Cam_NO'] = 175
        # print(PlateINFO)
        # plateOut['plateOut'] = PlateINFO
        #out = http_post()
        http_post1()
        #jdata = json.dumps(PlateINFO)
        #print("PlateINFO: ", jdata)
    # print(out)



