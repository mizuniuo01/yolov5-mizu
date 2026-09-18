import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))
import train

if __name__ == '__main__':
    train.run(
        imgsz=320,
        batch_size=-1,
        epochs=100,
        data='D:/Study/Python/YOLOv5/yolov5/data/myconfigs/testdata.yaml',
        weights='yolov5s.pt',
        device='0',
        workers=2,
        cache='ram'
    )
