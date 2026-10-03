import cv2
from core.video_reader import VideoReader


video = VideoReader("videos/sample.mp4")

print(video.get_info())

while True:
    frame = video.read()

    if frame is None:
        break

    cv2.imshow("Original Video", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

video.release()
cv2.destroyAllWindows()