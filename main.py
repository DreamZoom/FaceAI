import cv2  # opencv库

# 读取图片
image = cv2.imread('images/test2.jpg')

# face_model = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
# this model is goods
face_model = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
# 图片进行灰度处理
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# 人脸检测
faces = face_model.detectMultiScale(gray, 1.3, 5)
# 标记人脸
for (x, y, w, h) in faces:
    # 1.原始图片；2坐标点；3.矩形宽高 4.颜色值(RGB)；5.线框
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
# 显示图片窗口
cv2.imshow('人脸检测', image)
# 窗口暂停
cv2.waitKey(0)
# 销毁窗口资源
cv2.destroyAllWindows()
