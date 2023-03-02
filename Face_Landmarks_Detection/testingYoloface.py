from yoloface import face_analysis
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face = face_analysis()

while True:
    _, frame = cap.read()
    _, box, conf = face.face_detection(frame_arr=frame, frame_status=True, model='tiny')
    output_frame = face.show_output(frame, box, frame_status=True)
    cv2.imshow('frame', output_frame)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

