import cv2
import numpy as np


def plot_faces(frame, box, landmarks, name):
    # Draw a box around the face
    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

    # Draw landmarks
    landmarks = np.array(landmarks)
    for p in range(landmarks.shape[0]):
        cv2.circle(frame,
                   (int(landmarks[p, 0]), int(landmarks[p, 1])),
                   2, (0, 0, 255), -1, cv2.LINE_AA)

    # Draw a label with a name below the face
    cv2.rectangle(frame, (int(box[2]), int(box[3]) - 35), (int(box[2]), int(box[1])), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, str(name), (int(box[0]) + 6, int(box[1]) - 6), font, 1.0, (255, 255, 255), 1)

    return frame