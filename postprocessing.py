import cv2
import numpy as np
from PIL import Image, ImageTk


def plot_faces(frame, box, landmarks, name):
    # Draw a box around the face
    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
    temp_img = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
    try:
        face = cv2.resize(temp_img, dsize=(160, 160))
    except:
        print('small face resize failed')

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
    return frame, face


def gen_faces(faces):
    faces_img = []
    for face in faces:
        if len(faces_img) == 0:
            faces_img = face
        else:
            print(np.array(faces_img).shape)
            print(np.array(face).shape)
            # face = face[:, :, 0]
            print(np.array(face).shape)
            faces_img = np.concatenate((faces_img, face), axis=0)
    print(faces_img.shape)
    return faces_img
