from datetime import datetime
from ultralytics import YOLOv10
import cv2
import os

from ultralytics.create_vedio import vedio_writer
if __name__ == '__main__':
    # init
    video_path = '/home/huang/lift-splat-shoot/runs/prediction/vedio/lls.avi'
    save_path = 'runs/vedio'
    os.mkdir(save_path)
    cap = cv2.VideoCapture(video_path)
    i = 0
    # model = YOLOv10(config+'.yaml')
    model = YOLOv10('yolov8n.pt')
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, conf=0.5,
                                  tracker="bytetrack.yaml")
            cv2.imshow('result', results[0].plot())
            cv2.imwrite(os.path.join(
                save_path, '{}.jpg'.format(i)), results[0].plot())
            i += 1
            # Break the loop if 'q' is pressed
        else:
            break

    cap.release
    vedio_writer(save_path)
    #  model.export()
