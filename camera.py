from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face, fixed_image_standardization
import torch
import cv2
import numpy as np
from time import time

device = torch.device('cuda')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()
paths = np.load('paths.npy')
embeddings = np.load('embeddings.npy')

video_capture = cv2.VideoCapture(0)

frame_count = 0
offset = 30
boxes = None

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame_count += 1

    if frame_count == offset:
        start_time = time()

    if frame_count % 5 == 0:
        # update boxes only after 5 frames
        boxes, probs, points = mtcnn.detect(frame[...,::-1], landmarks=True)


    if type(boxes) != type(None):
        for box in boxes:
            (x1, y1, x2, y2) = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            resized = extract_face(frame[...,::-1], box).unsqueeze(0)
            resized = fixed_image_standardization(resized)
            with torch.no_grad():
                e = resnet(torch.Tensor(resized).cuda()).cpu().numpy()

            dists = np.linalg.norm(embeddings - e, axis=1)
            i = dists.argmin()
            print(paths[i], dists[i])

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time()

video_capture.release()
cv2.destroyAllWindows()

frame_count -= offset
print(f"FPS is {frame_count/(end_time-start_time)} over {frame_count} frames")