import os
import cv2
import numpy as np
import onnxruntime
import time

CLASSES=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'] #coco80类别

class YOLOV5():    #yolov5 onnx推理
    def __init__(self,onnxpath):
        self.onnx_session=onnxruntime.InferenceSession(onnxpath)
        self.input_name=self.get_input_name()
        self.output_name=self.get_output_name()
    def get_input_name(self):
        input_name=[]
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name
    def get_output_name(self):
        output_name=[]
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name
    def get_input_feed(self,img_tensor):
        input_feed={}
        for name in self.input_name:
            input_feed[name]=img_tensor
        return input_feed
    def inference(self,img_path):
        img=cv2.imread(img_path)   #读取图片
        or_img=cv2.resize(img,(640,640))
        img=or_img[:,:,::-1].transpose(2,0,1)  #BGR2RGB和HWC2CHW
        img=img.astype(dtype=np.float32)
        img/=255.0
        img=np.expand_dims(img,axis=0)
        input_feed=self.get_input_feed(img)
        pred=self.onnx_session.run(None,input_feed)[0]
        return pred,or_img

def pynms(dets, thresh): #非极大抑制
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1] #置信度从大到小排序（下标）

    while index.size > 0:
        i = index[0]
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # 计算相交面积
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # 当两个框不想交时x22 - x11或y22 - y11 为负数，
                                           # 两框不相交时把相交面积置0
        h = np.maximum(0, y22 - y11 + 1)  #

        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)#计算IOU

        idx = np.where(ious <= thresh)[0]  #IOU小于thresh的框保留下来
        index = index[idx + 1]  # 下标以1开始
        print(index)

    return keep


def xywh2xyxy(x):
    # [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def filter_box(org_box,conf_thres,iou_thres): #过滤掉无用的框
    org_box=np.squeeze(org_box) #删除为1的维度
    conf = org_box[..., 4] > conf_thres #删除置信度小于conf_thres的BOX
    # print(conf)
    box = org_box[conf == True]
    cls_cinf = box[..., 5:]
    cls = []
    for i in range(len(cls_cinf)):
        cls.append(int(np.argmax(cls_cinf[i])))
    all_cls = list(set(cls))     #删除重复的类别
    output = []
    for i in range(len(all_cls)):
        curr_cls = all_cls[i]
        curr_cls_box = []
        curr_out_box = []
        for j in range(len(cls)):
            if cls[j] == curr_cls:
                box[j][5] = curr_cls #将第6列元素替换为类别下标
                curr_cls_box.append(box[j][:6])   #当前类别的BOX
        curr_cls_box = np.array(curr_cls_box)
        curr_cls_box = xywh2xyxy(curr_cls_box)
        curr_out_box = pynms(curr_cls_box,iou_thres) #经过非极大抑制后输出的BOX下标
        for k in curr_out_box:
            output.append(curr_cls_box[k])  #利用下标取出非极大抑制后的BOX
    output = np.array(output)
    return output

def draw(image,box_data):  #画图
    boxes=box_data[...,:4].astype(np.int32) #取整方便画框
    scores=box_data[...,4]
    classes=box_data[...,5].astype(np.int32) #下标取整

    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)



if __name__=="__main__":
    # ----------------------omal----------------------
    onnx_path1='./yolov5s.onnx'
    model1=YOLOV5(onnx_path1)
    start1 = time.time()
    output1,or_img1=model1.inference('./bus.jpg')
    stop1 = time.time()
    print("omal time:",stop1 - start1)
    outbox=filter_box(output1,0.5,0.5)
    draw(or_img1,outbox)
    cv2.imshow('res1',or_img1)

    # ---------------------quant-------------------------
    onnx_path2='./model.quant.onnx'
    model2=YOLOV5(onnx_path2)
    start2 = time.time()
    output2,or_img2=model2.inference('./bus.jpg')
    stop2 = time.time()
    print("quant time:",stop2 - start2)
    outbox2=filter_box(output2,0.5,0.5)
    draw(or_img2,outbox)
    cv2.imshow('res2',or_img2)
    cv2.waitKey(0)
    cv2.destroyallwindows()