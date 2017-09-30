import cv2
import sys


# 加载分类器
car_haar = cv2.CascadeClassifier("car_judgement.xml")
plate_haar = cv2.CascadeClassifier("cascade.xml")

def CarIdentify(img):
    IsCar = {}
    # 把图像转为黑白图像
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_w, img_h = gray_img.shape[0], gray_img.shape[1]
    img_area = img_w * img_h
    # 检测图像总所有的车
    plates = plate_haar.detectMultiScale(gray_img, 1.08, 2, minSize=(36, 9),maxSize=(36*40, 9*40))
    cars = car_haar.detectMultiScale(gray_img)
    
    if len(cars) == 0 and len(plates) == 0:
        IsCar['isCar'] = None
    else:
        IsCar['isCar'] = True
        if len(cars) != 0:
            car_area = cars[0][2] * cars[0][3]
            IsCar['ratioCar'] = car_area / img_area
        if len(plates) != 0:
            plate_area = plates[0][2] * plates[0][3]
            IsCar['ratioPlate'] = plate_area / img_area        
    return IsCar
