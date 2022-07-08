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
            faces_img = np.concatenate((faces_img, face), axis=0)
    return faces_img

def getTracks(tracker, centroids):
    track_hist = {
            'Tracks': []
        }
    tracker.run(centroids)
    for i in range(len(tracker.measurements)):
        for track in tracker.tracks:
            x_hist = []
            y_hist = []
            colour_hist = []
            for t in range(0, track.x_hist.shape[0]):
                x_hist.append(track.x_hist[t, 0])
                y_hist.append(track.x_hist[t, 1])
                colour_hist.append(track.colour)
            track_hist['Tracks'].append({'x': x_hist, 'y': y_hist, 'colour': colour_hist})
    return track_hist

def drawTracks(img, tracks):
    for track in tracks:
        x = np.array(track.get('x'))
        y = np.array(track.get('y'))
        colour = track.get('colour')
        for i in range(x.shape[0]):
            cv2.circle(img, (int(x[i]), int(y[i])), 5, colour[i], 2)
    return img

def drawPeople(img, people):
    for person in people:
        cv2.rectangle(img, (int(person.get('x0')), int(person.get('y0'))), (int(person.get('x1')), int(person.get('y1'))), (0, 0, 255), 2)
    return img
