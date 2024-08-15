from ultralytics import YOLOv10


if __name__ == '__main__':
    model = YOLOv10('yolov10n.yaml')
    model.train(data='KITTI.yaml', optimizer='AdamW', lr0=6.5e-4, epochs=100, seed=1999, dnn=True, visualize=True,
                device=0)
    #  model.export()