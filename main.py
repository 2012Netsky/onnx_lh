import cv2
import onnx
from onnx import helper
import onnxruntime
import numpy as np

'''三、onnx模型推理'''
if __name__ == '__main__':

    ort_sess = onnxruntime.InferenceSession(
        'yolov5s.onnx')  # Create inference session using ort.InferenceSession

    # 加载图片
    img = cv2.imread("bus.jpg")
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((3, 640, 640))
    if len(img.shape) == 3:
        img = np.expand_dims(img, 0)

    outputs = ort_sess.run(None, {'images': img})  # 调用实例sess的run方法进行推理

    print(f"length of outputs = {len(outputs)}")

