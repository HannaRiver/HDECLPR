import time
import cv2
import numpy as np
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import json

import sys

fontC = ImageFont.truetype("platech.ttf", 14, 0);

def SimpleRecognizePlate(image):
    t0 = time.time()
    images = detectPlateRough(image,image.shape[0],top_bottom_padding_rate=0.1) # 车牌的粗定位
    res_set = []
    for j,plate in enumerate(images):
        plate, rect, origin_plate  = plate
        # plate = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
        plate  =cv2.resize(plate,(136,36*2))
        t1 = time.time()

        ptype = td_SimplePredict(plate)
        if ptype>0 and ptype<5:
            plate = cv2.bitwise_not(plate)

        image_rgb = findContoursAndDrawBoundingBox(plate)
        image_rgb = finemappingVertical(image_rgb)
        plate_hashname = verticalMappingToFolder(image_rgb)
        image_gray = cv2.cvtColor(image_rgb,cv2.COLOR_RGB2GRAY)

        # image_gray = horizontalSegmentation(image_gray)
        # cv2.imshow("image_gray",image_gray)
        # cv2.waitKey()

        cv2.imwrite("./tmp/"+str(j)+".jpg",image_gray)
        # cv2.imshow("image",image_gray)
        # cv2.waitKey(0)
        #print("校正",time.time() - t1,"s")
        # cv2.imshow("image,",image_gray)
        # cv2.waitKey(0)
        t2 = time.time()
        val = slidingWindowsEval(image_gray)
        # print val
        #print("分割和识别",time.time() - t2,"s")
        if len(val)==3:
            blocks, res, confidence = val
            if confidence/7>0.7:
                image = drawRectBox(image,rect,res)
                res_set.append(res)
                for i,block in enumerate(blocks):

                    block_ = cv2.resize(block,(25,25))
                    block_ = cv2.cvtColor(block_,cv2.COLOR_GRAY2BGR)
                    image[j * 25:(j * 25) + 25, i * 25:(i * 25) + 25] = block_
                    if image[j*25:(j*25)+25,i*25:(i*25)+25].shape == block_.shape:
                        pass
    return image,res_set

def drawRectBox(image,rect,addText):
    cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0,0, 255), 2,cv2.LINE_AA)
    cv2.rectangle(image, (int(rect[0]-1), int(rect[1])-16), (int(rect[0] + 80), int(rect[1])), (0, 0, 255), -1,
                  cv2.LINE_AA)

    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((int(rect[0]+1), int(rect[1]-16)), addText.encode('utf-8').decode(encoding="utf-8"), (255, 255, 255), font=fontC)
    imagex = np.array(img)

    return imagex

import scipy.ndimage.filters as f
import scipy

import time
import scipy.signal as l





from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K

K.set_image_dim_ordering('tf')


def Getmodel_tensorflow(nb_classes):
    # nb_classes = len(charset)
    img_rows, img_cols = 23, 23
    # number of convolutional filters to use
    nb_filters = 16
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    # x = np.load('x.npy')
    # y = np_utils.to_categorical(range(3062)*45*5*2, nb_classes)
    # weight = ((type_class - np.arange(type_class)) / type_class + 1) ** 3
    # weight = dict(zip(range(3063), weight / weight.mean()))  # 调整权重，高频字优先

    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(img_rows, img_cols,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.5))

    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model



def Getmodel_tensorflow_light(nb_classes):
    # nb_classes = len(charset)
    img_rows, img_cols = 23, 23
    # number of convolutional filters to use
    nb_filters = 8
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    # x = np.load('x.npy')
    # y = np_utils.to_categorical(range(3062)*45*5*2, nb_classes)
    # weight = ((type_class - np.arange(type_class)) / type_class + 1) ** 3
    # weight = dict(zip(range(3063), weight / weight.mean()))  # 调整权重，高频字优先

    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(img_rows, img_cols, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Convolution2D(nb_filters, nb_conv * 2, nb_conv * 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Flatten())
    model.add(Dense(32))
    # model.add(Dropout(0.25))

    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model




modelseg  = Getmodel_tensorflow_light(3)
model2seg  = Getmodel_tensorflow(3)

import os
# model.load_weights("char_judgement1.h5")
modelseg.load_weights('char_judgement1.h5')
model2seg.load_weights('char_judgement.h5')


modelseg = model2seg
def get_median(data):
    data = sorted(data)
    size = len(data)
    # print size
    # 判断列表长度为偶数
    if size % 2 == 0: 
        median = (data[size//2]+data[size//2-1])/2
        data[0] = median
    # 判断列表长度为奇数
    if size % 2 == 1:
        median = data[(size-1)//2]
        data[0] = median
    return data[0]
import time
def searchOptimalCuttingPoint(rgb,res_map,start,width_boundingbox,interval_range):
    t0  = time.time()
    #
    # for x in xrange(10):
    #     res_map = np.vstack((res_map,res_map[-1]))
    length = res_map.shape[0]
    refine_s = -2;

    if width_boundingbox>20:
        refine_s = -9
    score_list = []
    interval_big = int(width_boundingbox * 0.3)  #
    p = 0
    for zero_add in range(start,start+50,3):
        # for interval_small in xrange(-0,width_boundingbox/2):
            for i in range(-8,int(width_boundingbox/1)-8):
                for refine in range(refine_s,int(width_boundingbox/2+3)):
                    p1 = zero_add# this point is province
                    p2 = p1 + width_boundingbox +refine #
                    p3 = p2 + width_boundingbox + interval_big+i+1
                    p4 = p3 + width_boundingbox +refine
                    p5 = p4 + width_boundingbox +refine
                    p6 = p5 + width_boundingbox +refine
                    p7 = p6 + width_boundingbox +refine
                    if p7>=length:
                        continue
                    p1, p2, p3,p4,p5,p6,p7 = int(p1),int(p2),int(p3),int(p4),int(p5),int(p6),int(p7)
                    score = res_map[p1][2]*3 -(res_map[p3][1]+res_map[p4][1]+res_map[p5][1]+res_map[p6][1]+res_map[p7][1])+7
                    # print score
                    score_list.append([score,[p1,p2,p3,p4,p5,p6,p7]])
                    p+=1
    # print(p)

    score_list = sorted(score_list , key=lambda x:x[0])
    # for one in score_list[-1][1]:
    #     cv2.line(debug,(one,0),(one,36),(255,0,0),1)
    # #
    # cv2.imshow("one",debug)
    # cv2.waitKey(0)
    #
    # print("寻找最佳点",time.time()-t0)
    return score_list[-1]


sys.path.append('../')
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)

def refineCrop(sections,width=16):
    new_sections = []
    for section in sections:
        # cv2.imshow("section¡",section)

        # cv2.blur(section,(3,3),3)

        sec_center = np.array([section.shape[1]/2,section.shape[0]/2])
        binary_niblack = niBlackThreshold(section,17,-0.255)
        imagex, contours, hierarchy  = cv2.findContours(binary_niblack,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        boxs = []
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)

            ratio = w/float(h)
            if ratio<1 and h>36*0.4 and y<16:
                box = [x,y,w,h]

                boxs.append([box,np.array([x+w/2,y+h/2])])

        dis_ = np.array([ ((one[1]-sec_center)**2).sum() for one in boxs])
        if len(dis_)==0:
            kernal = [0, 0, section.shape[1], section.shape[0]]
        else:
            kernal = boxs[dis_.argmin()][0]

        center_c  = (kernal[0]+kernal[2]/2,kernal[1]+kernal[3]/2)
        w_2 = int(width/2)
        h_2 = kernal[3]/2

        if center_c[0] - w_2< 0:
            w_2 = center_c[0]
        new_box = [center_c[0] - w_2,kernal[1],width,kernal[3]]
        # print new_box[2]/float(new_box[3])
        if new_box[2]/float(new_box[3])>0.5:
            # print "异常"
            h = int((new_box[2]/0.35 )/2)
            if h>35:
                h = 35
            new_box[1] = center_c[1]- h
            if new_box[1]<0:
                new_box[1] = 1

            new_box[3] = h*2

        section = section[int(new_box[1]):int(new_box[1]+new_box[3]), int(new_box[0]):int(new_box[0]+new_box[2])]
        # cv2.imshow("section",section)
        # cv2.waitKey(0)
        new_sections.append(section)
        # print new_box


    return new_sections





def slidingWindowsEval(image):
    windows_size = 16;
    stride = 1
    height= image.shape[0]
    t0 = time.time()
    data_sets = []

    for i in range(0,image.shape[1]-windows_size+1,stride):
        data = image[0:height,i:i+windows_size]
        data = cv2.resize(data,(23,23))
        # cv2.imshow("image",data)
        data = cv2.equalizeHist(data)
        data = data.astype(np.float)/255
        data=  np.expand_dims(data,3)
        data_sets.append(data)

    res = modelseg.predict(np.array(data_sets))
    # print("分割",time.time() - t0)

    pin = res
    p = 1 -  (res.T)[1]
    p = f.gaussian_filter1d(np.array(p,dtype=np.float),3)
    lmin = l.argrelmax(np.array(p),order = 3)[0]
    interval = []
    for i in range(len(lmin)-1):
        interval.append(lmin[i+1]-lmin[i])

    if(len(interval)>3):
        mid  = get_median(interval)
    else:
        return []
    pin = np.array(pin)
    res =  searchOptimalCuttingPoint(image,pin,0,mid,3)

    cutting_pts = res[1]
    last =  cutting_pts[-1] + mid
    if last < image.shape[1]:
        cutting_pts.append(last)
    else:
        cutting_pts.append(image.shape[1]-1)
    name = ""
    confidence =0.00
    seg_block = []
    for x in range(1,len(cutting_pts)):
        if x != len(cutting_pts)-1 and x!=1:
            section = image[0:36,cutting_pts[x-1]-2:cutting_pts[x]+2]
        elif  x==1:
            c_head = cutting_pts[x - 1]- 2
            if c_head<0:
                c_head=0
            c_tail = cutting_pts[x] + 2
            section = image[0:36, c_head:c_tail]
        elif x==len(cutting_pts)-1:
            end = cutting_pts[x]
            diff = image.shape[1]-end
            c_head = cutting_pts[x - 1]
            c_tail = cutting_pts[x]
            if diff<7 :
                section = image[0:36, int(c_head-5):int(c_tail+5)]
            else:
                diff-=1
                section = image[0:36, int(c_head - diff):int(c_tail + diff)]
        elif  x==2:
            section = image[0:36, int(cutting_pts[x - 1] - 3):int(cutting_pts[x-1]+ mid)]
        else:
            section = image[0:36,int(cutting_pts[x-1]):int(cutting_pts[x])]
        seg_block.append(section)
    refined = refineCrop(seg_block,mid-1)

    t0 = time.time()
    for i,one in enumerate(refined):
        res_pre = cRP_SimplePredict(one, i )
        # cv2.imshow(str(i),one)
        # cv2.waitKey(0)
        confidence+=res_pre[0]
        name+= res_pre[1]
    # print("字符识别",time.time() - t0)

    return refined,name,confidence

from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)


def niBlackThreshold(  src,  blockSize,  k,  binarizationMethod=0):
    mean = cv2.boxFilter(src,cv2.CV_32F,(blockSize, blockSize),borderType=cv2.BORDER_REPLICATE)
    sqmean = cv2.sqrBoxFilter(src, cv2.CV_32F, (blockSize, blockSize), borderType = cv2.BORDER_REPLICATE)
    variance = sqmean - (mean*mean)
    stddev  = np.sqrt(variance)
    thresh = mean + stddev * float(-k)
    thresh = thresh.astype(src.dtype)
    k = (src>thresh)*255
    k = k.astype(np.uint8)
    return k


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K

K.set_image_dim_ordering('tf')


import cv2
import numpy as np



index = {u"京": 0, u"沪": 1, u"津": 2, u"渝": 3, u"冀": 4, u"晋": 5, u"蒙": 6, u"辽": 7, u"吉": 8, u"黑": 9, u"苏": 10, u"浙": 11, u"皖": 12,
         u"闽": 13, u"赣": 14, u"鲁": 15, u"豫": 16, u"鄂": 17, u"湘": 18, u"粤": 19, u"桂": 20, u"琼": 21, u"川": 22, u"贵": 23, u"云": 24,
         u"藏": 25, u"陕": 26, u"甘": 27, u"青": 28, u"宁": 29, u"新": 30, u"0": 31, u"1": 32, u"2": 33, u"3": 34, u"4": 35, u"5": 36,
         u"6": 37, u"7": 38, u"8": 39, u"9": 40, u"A": 41, u"B": 42, u"C": 43, u"D": 44, u"E": 45, u"F": 46, u"G": 47, u"H": 48,
         u"J": 49, u"K": 50, u"L": 51, u"M": 52, u"N": 53, u"P": 54, u"Q": 55, u"R": 56, u"S": 57, u"T": 58, u"U": 59, u"V": 60,
         u"W": 61, u"X": 62, u"Y": 63, u"Z": 64,u"港":65,u"学":66 ,u"O":67 ,u"使":68,u"警":69,u"澳":70,u"挂":71};

chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
             "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P",
         "Q", "R", "S", "T", "U", "V", "W", "X",
             "Y", "Z","港","学","O","使","警","澳","挂"];



def Getmodel_tensorflow(nb_classes):
    # nb_classes = len(charset)

    img_rows, img_cols = 23, 23
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    # x = np.load('x.npy')
    
    # y = np_utils.to_categorical(range(3062)*45*5*2, nb_classes)
    # weight = ((type_class - np.arange(type_class)) / type_class + 1) ** 3
    # weight = dict(zip(range(3063), weight / weight.mean()))  # 调整权重，高频字优先

    model = Sequential()
    model.add(Convolution2D(32, 5, 5,
                            border_mode='valid',
                            input_shape=(img_rows, img_cols,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(512, 3, 3))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model




def Getmodel_ch(nb_classes):
    # nb_classes = len(charset)

    img_rows, img_cols = 23, 23
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    # x = np.load('x.npy')
    # y = np_utils.to_categorical(range(3062)*45*5*2, nb_classes)
    # weight = ((type_class - np.arange(type_class)) / type_class + 1) ** 3
    # weight = dict(zip(range(3063), weight / weight.mean()))  # 调整权重，高频字优先

    model = Sequential()
    model.add(Convolution2D(32, 5, 5,
                            border_mode='valid',
                            input_shape=(img_rows, img_cols,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(512, 3, 3))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(756))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model



modelrec  = Getmodel_tensorflow(65)
#构建网络

model_chrec = Getmodel_ch(31)

model_chrec.load_weights('char_chi_sim.h5')
modelrec.load_weights("char_rec.h5")


def cRP_SimplePredict(image,pos):
    image = cv2.resize(image, (23, 23))
    image = cv2.equalizeHist(image)
    image = image.astype(np.float) / 255
    image -= image.mean()
    image = np.expand_dims(image, 3)
    if pos!=0:
        res = np.array(modelrec.predict(np.array([image]))[0])
    else:
        res = np.array(model_chrec.predict(np.array([image]))[0])

    zero_add = 0 ;

    if pos==0:
        res = res[:31]
    elif pos==1:
        res = res[31+10:65]
        zero_add = 31+10
    else:
        res = res[31:]
        zero_add = 31

    max_id = res.argmax()


    return res.max(),chars[int(max_id+zero_add)],max_id+zero_add


import hashlib

def verticalMappingToFolder(image):
    name = hashlib.md5(image.data).hexdigest()[:8]
    # print(name)
    cv2.imwrite("./tmp/"+name+".png",image)
    return name


from keras.layers import Conv2D, Input,MaxPool2D, Reshape,Activation,Flatten, Dense
from keras.models import Model, Sequential
from keras.layers.advanced_activations import PReLU
from keras.optimizers import adam
import numpy as np

import cv2

def getModel():


    input = Input(shape=[12, 50, 3])  # change this shape to [None,None,3] to enable arbitraty shape input
    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=2)(x)
    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
    x = Flatten()(x)
    output = Dense(2)(x)
    output = PReLU(name='prelu4')(output)
    model = Model([input], [output])
    return model

modelv = getModel()
modelv.load_weights("model12.h5")


def finemappingVertical(image):
    resized = cv2.resize(image,(50,12))
    resized = resized.astype(np.float)/255
    res= modelv.predict(np.array([resized]))[0]
    res  =res*image.shape[1]
    res = res.astype(np.int)
    image = image[0:35,res[0]+4:res[1]]
    image = cv2.resize(image, (int(136), int(36)))
    return image


def fitLine_ransac(pts,zero_add = 0 ):
    if len(pts)>=2:
        [vx, vy, x, y] = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((136- x) * vy / vx) + y)
        return lefty+30+zero_add,righty+30+zero_add
    return 0,0

#精定位算法
def findContoursAndDrawBoundingBox(image_rgb):


    line_upper  = [];
    line_lower = [];

    line_experiment = []
    grouped_rects = []
    gray_image = cv2.cvtColor(image_rgb,cv2.COLOR_BGR2GRAY)

    # for k in np.linspace(-1.5, -0.2,10):
    for k in np.linspace(-50, 0, 15):

        # thresh_niblack = threshold_niblack(gray_image, window_size=21, k=k)
        # binary_niblack = gray_image > thresh_niblack
        # binary_niblack = binary_niblack.astype(np.uint8) * 255

        binary_niblack = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,17,k)
        # cv2.imshow("image1",binary_niblack)
        # cv2.waitKey(0)
        imagex, contours, hierarchy = cv2.findContours(binary_niblack.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            bdbox = cv2.boundingRect(contour)
            if (bdbox[3]/float(bdbox[2])>0.7 and bdbox[3]*bdbox[2]>100 and bdbox[3]*bdbox[2]<1200) or (bdbox[3]/float(bdbox[2])>3 and bdbox[3]*bdbox[2]<100):
                # cv2.rectangle(rgb,(bdbox[0],bdbox[1]),(bdbox[0]+bdbox[2],bdbox[1]+bdbox[3]),(255,0,0),1)
                line_upper.append([bdbox[0],bdbox[1]])
                line_lower.append([bdbox[0]+bdbox[2],bdbox[1]+bdbox[3]])

                line_experiment.append([bdbox[0],bdbox[1]])
                line_experiment.append([bdbox[0]+bdbox[2],bdbox[1]+bdbox[3]])
                # grouped_rects.append(bdbox)

    rgb = cv2.copyMakeBorder(image_rgb,30,30,0,0,cv2.BORDER_REPLICATE)
    leftyA, rightyA = fitLine_ransac(np.array(line_lower),3)
    rows,cols = rgb.shape[:2]

    # rgb = cv2.line(rgb, (cols - 1, rightyA), (0, leftyA), (0, 0, 255), 1,cv2.LINE_AA)

    leftyB, rightyB = fitLine_ransac(np.array(line_upper),-3)

    rows,cols = rgb.shape[:2]

    # rgb = cv2.line(rgb, (cols - 1, rightyB), (0, leftyB), (0,255, 0), 1,cv2.LINE_AA)
    pts_map1  = np.float32([[cols - 1, rightyA], [0, leftyA],[cols - 1, rightyB], [0, leftyB]])
    pts_map2 = np.float32([[136,36],[0,36],[136,0],[0,0]])
    mat = cv2.getPerspectiveTransform(pts_map1,pts_map2)
    image = cv2.warpPerspective(rgb,mat,(136,36),flags=cv2.INTER_CUBIC)
    image = fastDeskew(image)

    return image


import math

from scipy.ndimage import filters


def angle(x,y):
    return int(math.atan2(float(y),float(x))*180.0/3.1415);

def h_rot(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale

    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)

    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))

    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

def v_rot(img,angel,shape,max_angel):

    size_o = [shape[1],shape[0]]

    size = (shape[1]+ int(shape[0]*np.cos((float(max_angel )/180) * 3.14)),shape[0])


    interval = abs( int( np.sin((float(angel) /180) * 3.14)* shape[0]));

    pts1 = np.float32([[0,0]         ,[0,size_o[1]],[size_o[0],0],[size_o[0],size_o[1]]])
    if(angel>0):

        pts2 = np.float32([[interval,0],[0,size[1]  ],[size[0],0  ],[size[0]-interval,size_o[1]]])
    else:
        pts2 = np.float32([[0,0],[interval,size[1]  ],[size[0]-interval,0  ],[size[0],size_o[1]]])

    M  = cv2.getPerspectiveTransform(pts1,pts2);
    dst = cv2.warpPerspective(img,M,size);
    return dst

def skew_detection(image_gray):
    h, w = image_gray.shape[:2]
    eigen = cv2.cornerEigenValsAndVecs(image_gray,12, 5) # 计算图像块的特征值和特征向量用于角点检测
    angle_sur = np.zeros(180,np.uint);
    eigen = eigen.reshape(h, w, 3, 2)
    flow = eigen[:,:,2]
    vis = image_gray.copy()
    vis[:] = (192 + np.uint32(vis)) / 2
    d = 12
    points =  np.dstack( np.mgrid[d/2:w:d, d/2:h:d] ).reshape(-1, 2)
    # print(points)
    for x, y in points:
        x, y = int(x), int(y)
        # print(flow[y, x]*d)
        vx, vy = np.int32(flow[y, x]*d)
        # cv2.line(rgb, (x-vx, y-vy), (x+vx, y+vy), (0, 355, 0), 1, cv2.LINE_AA)
        ang = angle(vx,vy);
        angle_sur[(ang+180)%180] +=1;
    # torr_bin = 30
    angle_sur = angle_sur.astype(np.float)
    angle_sur = (angle_sur-angle_sur.min())/(angle_sur.max()-angle_sur.min())
    angle_sur = filters.gaussian_filter1d(angle_sur,5)
    skew_v_val =  angle_sur[20:180-20].max();
    skew_v = angle_sur[30:180-30].argmax() + 30;
    skew_h_A = angle_sur[0:30].max()
    skew_h_B = angle_sur[150:180].max()
    skew_h = 0;
    if (skew_h_A > skew_v_val*0.3 or skew_h_B > skew_v_val*0.3):
        if skew_h_A>=skew_h_B:
            skew_h = angle_sur[0:20].argmax()
        else:
            skew_h = - angle_sur[160:180].argmax()
    return skew_h,skew_v

def fastDeskew(image):
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    skew_h,skew_v = skew_detection(image_gray)

    # print("校正角度 h ",skew_h,"v",skew_v)

    deskew = v_rot(image,int((90-skew_v)*1.5),image.shape,60)
    return deskew

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K

K.set_image_dim_ordering('tf')


import cv2
import numpy as np


plateType  = [u"蓝牌",u"单层黄牌",u"新能源车牌",u"白色",u"黑色-港澳"]
def Getmodel_tensorflow(nb_classes):
    # nb_classes = len(charset)

    img_rows, img_cols = 9, 34
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    # x = np.load('x.npy')
    # y = np_utils.to_categorical(range(3062)*45*5*2, nb_classes)
    # weight = ((type_class - np.arange(type_class)) / type_class + 1) ** 3
    # weight = dict(zip(range(3063), weight / weight.mean()))  # 调整权重，高频字优先

    model = Sequential()
    model.add(Convolution2D(16, 5, 5,
                            border_mode='valid',
                            input_shape=(img_rows, img_cols,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

model = Getmodel_tensorflow(5)
model.load_weights("plate_type.h5")
def td_SimplePredict(image):
    image = cv2.resize(image, (34, 9))
    image = image.astype(np.float) / 255
    res = np.array(model.predict(np.array([image]))[0])
    return res.argmax()


watch_cascade = cv2.CascadeClassifier('cascade.xml')

def computeSafeRegion(shape,bounding_rect):
    top = bounding_rect[1] # y
    bottom  = bounding_rect[1] + bounding_rect[3] # y +  h
    left = bounding_rect[0] # x
    right =   bounding_rect[0] + bounding_rect[2] # x +  w

    min_top = 0
    max_bottom = shape[0]
    min_left = 0
    max_right = shape[1]

    # print "computeSateRegion input shape",shape
    if top < min_top:
        top = min_top
        # print "tap top 0"
    if left < min_left:
        left = min_left
        # print "tap left 0"

    if bottom > max_bottom:
        bottom = max_bottom
        #print "tap max_bottom max"
    if right > max_right:
        right = max_right
        #print "tap max_right max"

    # print "corr",left,top,right,bottom
    return [left,top,right-left,bottom-top]


def cropped_from_image(image,rect):
    x, y, w, h = computeSafeRegion(image.shape,rect)
    return image[y:y+h,x:x+w]


def detectPlateRough(image_gray,resize_h = 720,en_scale =1.08 ,top_bottom_padding_rate = 0.05):
    # print(image_gray.shape)

    if top_bottom_padding_rate>0.2:
        # print("error:top_bottom_padding_rate > 0.2:",top_bottom_padding_rate)
        exit(1)

    height = image_gray.shape[0]
    padding = int(height*top_bottom_padding_rate)
    scale = image_gray.shape[1]/float(image_gray.shape[0])

    image = cv2.resize(image_gray, (int(scale*resize_h), resize_h))

    image_color_cropped = image[padding:resize_h-padding,0:image_gray.shape[1]]

    image_gray = cv2.cvtColor(image_color_cropped,cv2.COLOR_RGB2GRAY)

    watches = watch_cascade.detectMultiScale(image_gray, en_scale, 2, minSize=(36, 9),maxSize=(36*40, 9*40))

    cropped_images = []
    for (x, y, w, h) in watches:
        cropped_origin = cropped_from_image(image_color_cropped, (int(x), int(y), int(w), int(h)))
        x -= w * 0.14
        w += w * 0.28
        y -= h * 0.6
        h += h * 1.1;

        cropped = cropped_from_image(image_color_cropped, (int(x), int(y), int(w), int(h)))


        cropped_images.append([cropped,[int(x), int(y+padding), int(w), int(h)],cropped_origin])
    return cropped_images

def RecognizePlateJson(image):
    
    images = detectPlateRough(image,image.shape[0],top_bottom_padding_rate=0.1)

    jsons = []

    for j,plate in enumerate(images):


        plate,rect,origin_plate =plate


        cv2.imwrite("./tmp/"+str(j)+"_rough.jpg",plate)

        # print "车牌类型:",ptype
        # plate = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
        plate  =cv2.resize(plate,(136,int(36*2.5)))
        t1 = time.time()


        ptype = td_SimplePredict(plate)
        if ptype>0 and ptype<5:
            plate = cv2.bitwise_not(plate)
        # demo = verticalEdgeDetection(plate)

        image_rgb = findContoursAndDrawBoundingBox(plate)
        image_rgb = finemappingVertical(image_rgb)
        plate_hashname = verticalMappingToFolder(image_rgb)
        # print time.time() - t1,"校正"

        image_gray = cv2.cvtColor(image_rgb,cv2.COLOR_BGR2GRAY)


        cv2.imwrite("./tmp/"+str(j)+".jpg",image_gray)
        # image_gray = horizontalSegmentation(image_gray)


        t2 = time.time()
        val = slidingWindowsEval(image_gray)
        if len(val)==3:
            blocks, res, confidence = val
            if confidence/7>0.7:
                image = drawRectBox(image,rect,res)
            for i,block in enumerate(blocks):

                block_ = cv2.resize(block,(25,25))
                block_ = cv2.cvtColor(block_,cv2.COLOR_GRAY2BGR)
                image[j * 25:(j * 25) + 25, i * 25:(i * 25) + 25] = block_
                if image[j*25:(j*25)+25,i*25:(i*25)+25].shape == block_.shape:
                    pass

            plate_name =res
            res_json = {}
            if confidence/7>0.7:
                res_json["Plate_Number"] = plate_name
                res_json["Plate_Type"] = plateType[ptype]
                res_json["Confidence"] = confidence/7;
                res_json["Plat_x"] = int(rect[0])
                res_json["Plat_y"] = int(rect[1])
                res_json["Plat_w"] = int(rect[2])
                res_json["Plat_h"] = int(rect[3])
                # res_json["Image_path"] = plate_hashname
                # print "车牌:",res,"置信度:",confidence/7
                jsons.append(res_json)


            else:
                pass
                # print "不确定的车牌:", res, "置信度:", confidence
    # print(jsons)
    # print(json.dumps(jsons,ensure_ascii=False,encoding="gb2312"))

    #return json.dumps(jsons,ensure_ascii=False,encoding="gb2312")
    return jsons





def clpr(path):
    image = cv2.imread(path)
    plateOut = RecognizePlateJson(image)
    for plate in plateOut:
        plate["Image_path"] = path
    return plateOut
