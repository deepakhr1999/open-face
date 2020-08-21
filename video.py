from facenet_pytorch import MTCNN, InceptionResnetV1
import matplotlib.pyplot as plt

import cv2
import numpy as np
from time import time
import torch

device = torch.device('cuda')
mtcnn = MTCNN(keep_all=True, device=device)
# resnet = InceptionResnetV1(pretrained='vggface2').eval()


video_capture = cv2.VideoCapture('video.mp4')
start_time = time()
frame_count = 0
while video_capture.isOpened():
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame_count += 1
    
    if not ret:
        break
    boxes, probs, points = mtcnn.detect(frame[...,::-1], landmarks=True)

    for box in boxes:
        (x1, y1, x2, y2) = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # cv2.imshow('Video', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

end_time = time()

video_capture.release()
cv2.destroyAllWindows()

print(f"FPS is {frame_count/(end_time-start_time)} over {frame_count} frames")