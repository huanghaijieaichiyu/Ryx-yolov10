from datetime import datetime
from ultralytics import YOLOv10
import cv2
import os
if __name__ == '__main__':
    # init
    video_path = '/home/huang/lift-splat-shoot/runs/prediction/vedio/lls.avi'
    video_output = os.path.join('runs', 'result.avi')
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    video = cv2.VideoWriter(
        video_output, fourcc, 24, (480, 640))

    # model = YOLOv10(config+'.yaml')
    model = YOLOv10('yolov8n.pt')
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, conf=0.3,
                                  tracker="bytetrack.yaml")
            video.write(results[0].plot())
            cv2.imshow("YOLOv8 Tracking", results[0].plot())

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) == 27:
                break
        else:
            break

    cap.release
    video.release
    #  model.export()
