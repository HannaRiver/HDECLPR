#!/usr/bin/env python3
import clpr as clpr
import car_identify as caride
import cv2

import urllib
import json
import urllib.request


def http_post(url='127.0.0.1'):
    values = PlateINFO
    jdata = json.dumps(values)
    req = urllib.request.Request(url, jdata) # 生成页面请求的完整数据
    response = urllib.request.urlopen(req) # 发送页面请求
    return response.read()

if __name__ == '__main__':
    img_path = 'JingH99999.jpg'
    image = cv2.imread(img_path)
    # CarINFO = {}
    CarINFO = caride.CarIdentify(image)
    if CarINFO['isCar']:
        PlateINFOs = clpr.clpr(img_path)
        PlateINFO = PlateINFO[0]    
    http_post()
