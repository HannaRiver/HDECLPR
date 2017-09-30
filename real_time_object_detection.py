# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse # 命令行参数解析模块
import imutils
import time
import cv2


# 解析命令行参数
ap = argparse.ArgumentParser()
# Caffe protetext文件路径
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototext file")
# 预训练模型的路径
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
# 过滤弱检测的最小概率阈值，默认值为20%
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# 初始化类列表和颜色集
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
"bus", "car", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
"person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# 加载自己的模型，并设置自己的视频流
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
# 启动 VideoStream
vs = VideoStream(src=1).start()
# 等待相机启动
time.sleep(2.0)
# 最后开始每秒帧数计算
fps = FPS().start()

# 遍历每一帧
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    # 从视频流中读取一帧
    frame = vs.read()
    # 调整它的大小
    frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and predictions
    # 设置 blob 为神经网络的输入
    net.setInput(blob)
    # 通过 net 传递输入
    detections = net.forward()

    # 我们已经在输入帧中检测到了目标，现在是时候看看置信度的值，以判断我们能否在目标周围绘制边界框和标签了
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., pribability) associated with the prediction
        # 首先我们提取 confidence 值
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is greater than
        # minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y) - coordinates of the bounding box for the object
            # 提取类标签索引
            idx = int(detections[0, 0, i, 1])
            # 计算检测到的目标的坐标
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            # 提取边界框的 (x, y) 坐标，之后将用于绘制矩形和文本
            (startX, startY, endX, endY) = box.astype("int")

            # draw the prediction on the frame
            # 构建一个文本 label，包含 CLASS 名称和 confidence
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            # 使用类颜色和之前提取的 (x, y) 坐标在物体周围绘制彩色矩形
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            # 使用刚才计算出的 y 值将彩色文本置于帧上
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        # 帧捕捉循环剩余的步骤还包括：（1）展示帧；（2）检查 quit 键；（3）更新 fps 计数器：
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        # 同时检查「q」键（代表「quit」）是否按下。如果已经按下，则我们退出帧捕捉循环
        if key == ord("q"):
            break

        # 最后更新 fps 计数器
        fps.update()
    # stop the timer and display FPS information
    fps.stop()
    # 每秒帧数的信息向终端输出
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()












