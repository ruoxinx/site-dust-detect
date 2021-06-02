# A Keras implementation of YOLOv3 (Tensorflow backend) for dust detection inspired by [bubbliiiing/yolo3-keras]

from PIL import Image
from yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO()
    img = Image.open('../examples/01.jpg')
    result = yolo.detect_image(img)
    result.show()