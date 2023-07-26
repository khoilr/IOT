from datetime import datetime
import cv2

from face_detector import YoloV5FaceDetector
from face_detector.utils import draw_boundary, draw_points, draw_text

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("video.mp4")
face_detector = YoloV5FaceDetector()

while cap.isOpened():
    success, frame = cap.read()
    # cvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # boundaries, five_points, confidences, nimgs = face_detector.detect_in_image(cvImg)

    # # save nimgs to disk
    # if nimgs is not None and len(nimgs) != 0:
    #     for index, nimg in enumerate(nimgs):
    #         cv2.imwrite("nimg_{}.png".format(int(datetime.now().timestamp())), cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR))

    # for index, boundary in enumerate(boundaries):
    #     # Draw the rectangle boundary
    #     draw_boundary(
    #         img=cvImg,
    #         boundaries=boundary,
    #         color=(0, 255, 0),  # Green
    #         thickness=2,
    #     )

    #     if confidences is not None and len(confidences) != 0:
    #         confidence = confidences[index]
    #         # Draw the confidence score as text
    #         draw_text(
    #             img=cvImg,
    #             text="{:.4f}".format(confidence),
    #             org=(int(boundary[0]), int(boundary[1])),
    #             font_scale=0.5,
    #             color=(0, 0, 255),
    #             thickness=1,
    #         )  # Red color, font scale=0.5, thickness=1

    #     if five_points is not None and len(five_points) != 0:
    #         five_point = five_points[index]
    #         # Draw the points as circles
    #         if len(five_point.shape) == 2:
    #             draw_points(
    #                 img=cvImg,
    #                 points=five_point,
    #                 color=(255, 0, 0),  # Blue
    #                 radius=2,
    #             )
    #         # Draw the points as individual circles
    #         else:
    #             draw_points(
    #                 img=cvImg,
    #                 points=[(five_point[i], five_point[i + 1]) for i in range(0, len(five_point), 2)],
    #                 color=(255, 0, 0),  # Blue
    #                 radius=2,
    #             )

    # # Show the image
    # cv2.imshow("frame", cv2.cvtColor(cvImg, cv2.COLOR_RGB2BGR))

    cv2.imshow("frame", frame)

    # Press "q" to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
