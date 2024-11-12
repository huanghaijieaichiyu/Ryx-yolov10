from datetime import datetime
import cv2
from numpy import source
import torch
import numpy as np
from ultralytics import YOLOv10
from ultralytics.trackers import track
from collections import defaultdict

if __name__ == '__main__':
    model = YOLOv10('runs/detect/iRMB_KITTI_24-08-18_10-22-08/weights/best.pt')
    # Open the video file
    video_path = 'runs/pred/result.avi'
    cap = cv2.VideoCapture(video_path)
    path = 'runs'
    video_output = os.path.join(path, 'result.avi')
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    video = cv2.VideoWriter(
        video_output, fourcc, 30, (480, 640))
    track_history = defaultdict(lambda: [])
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True,
                              tracker="bytetrack.yaml")
        if results[0].boxes.id is not None:
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist(
            )

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    annotated_frame,
                    [points],
                    isClosed=False,
                    color=(230, 230, 230),
                    thickness=10,
                )
                cv2.imshow("YOLOv8 Tracking", annotated_frame)
                video.write(annotated_frame)
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break

    # Release the video capture object and close the display window
    cap.release()
    video.release()
    cv2.destroyAllWindows()
