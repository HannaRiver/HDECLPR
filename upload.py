#!/usr/bin/env python3
import clpr as clpr
import car_identify as caride
import cv2
import imageio

import urllib
import json
import urllib.request


def http_post(url='http://www.parking.com/'):
    values = PlateINFO
    jdata = json.dumps(values)
    jdata = "plateOut="+ jdata
    jdata = jdata.encode('utf-8')
    req = urllib.request.Request(url, jdata) # 生成页面请求的完整数据
    response = urllib.request.urlopen(req) # 发送页面请求
    return response.read()

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
        # plateOut['plateOut'] = PlateINFO
        out = http_post()
        jdata = json.dumps(PlateINFO)
        #print("PlateINFO: ", jdata)
    
    print(out)