import os
from ultralytics import YOLOv10


if __name__ == '__main__':
    model = YOLOv10('iRMB.yaml')
    model.train(data='KITTI.yaml', optimizer='AdamW', lr0=6.5e-4, epochs=100)
    #  model.export()