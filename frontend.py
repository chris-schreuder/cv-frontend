import io
import json
import os

from time import time, sleep
import time
import requests

from flask import Flask, render_template, Response, request

import cv2
import numpy as np

import postprocessing

ENDPOINT_recognition = 'http://127.0.0.1:8000/api/facial_recognition/inference/'
ENDPOINT_tracking = 'http://127.0.0.1:8000/api/people_detection/inference/'

cap = None
video = None
currentframe = None
response = []

identified = []
unidentified = []
prev = 0
prevIdent = 0
prevUnident = 0

def activate_stream():
    global cap
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'vp80')

    global currentframe
    currentframe = 0

def deactivate_stream():
    global cap
    cap = None
    global video
    video = None
    global currentframe
    currentframe = None

# Initialise Flask
app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    deactivate_stream()
    return render_template('index.html')

@app.route('/stream')
def stream():
    if not cap:
        activate_stream()
    return render_template('stream.html')

@app.route('/recognition')
def recognition():
    if not cap == None:
        activate_stream()
    return render_template('recognition.html')

@app.route('/tracking')
def tracking():
    if not cap == None:
        activate_stream()
    return render_template('tracking.html')

@app.route('/return')
def main_site():
    deactivate_stream()
    main_url = request.host_url.replace("8001","8000").replace('return','/')
    return render_template('return.html', main_url=main_url)

def gen():
    try:
        cap.isOpened()
    except AttributeError:
        activate_stream()
    while (cap.isOpened()):
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        if ret:
            img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
            frame = cv2.imencode('.png', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')
        else:
            break

def recognition_gen():
    global identified
    global unidentified
    global prev
    global response
    frame_rate = 5
    try:
        cap.isOpened()
    except AttributeError:
        activate_stream()
    while (cap.isOpened()):
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        if ret:
            try:
                time_elapsed = time.time() - prev
                if time_elapsed > 1. / frame_rate:
                    prev = time.time()
                    _, jpeg = cv2.imencode('.jpg', img)
                    payload = {'image': jpeg.tobytes()}
                    response = requests.post(ENDPOINT_recognition,
                                             files=payload).json()
                matched = response.get('FaceMatches')
                unmatched = response.get('UnmatchedFaces')
                identified = []
                unidentified = []
                for i in matched:
                    img, cropped = postprocessing.plot_faces(frame=img, box=i.get('Box'), landmarks=i.get('Landmarks'), name=i.get('Name'))
                    identified.append(cropped)
                for i in unmatched:
                    img, cropped = postprocessing.plot_faces(frame=img, box=i.get('Box'), landmarks=i.get('Landmarks'), name=i.get('Name'))
                    unidentified.append(cropped)
            except Exception as e:
                print(str(e))
            img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
            frame = cv2.imencode('.png', img)[1].tobytes()
            #print('before send')
            yield (b'--frame\r\n'b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')
            #print('after send')
        else:
            break

def getIdentified():
    global identified
    global prevIdent
    frame_rate = 0.5
    try:
        cap.isOpened()
    except AttributeError:
        activate_stream()
    while (cap.isOpened()):
        time_elapsed = time.time() - prevIdent
        if time_elapsed > 1. / frame_rate:
            prevIdent = time.time()
            if len(identified) > 0:
                try:
                    img = postprocessing.gen_faces(identified)
                    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                    img = cv2.imencode('.png', img)[1].tobytes()
                    yield (b'--frame\r\n'b'Content-Type: image/png\r\n\r\n' + img + b'\r\n')
                except Exception as e:
                    print(str(e))
            else:
                blank_image = np.zeros((160, 160, 3), np.uint8)
                img = cv2.resize(blank_image, (0, 0), fx=0.5, fy=0.5)
                img = cv2.imencode('.png', img)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/png\r\n\r\n' + img + b'\r\n')

def getUnidentified():
    global unidentified
    global prevUnident
    frame_rate = 0.5
    try:
        cap.isOpened()
    except AttributeError:
        activate_stream()
    while (cap.isOpened()):
        time_elapsed = time.time() - prevIdent
        if time_elapsed > 1. / frame_rate:
            prevIdent = time.time()
            if len(unidentified) > 0:
                try:
                    img = postprocessing.gen_faces(unidentified)
                    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                    img = cv2.imencode('.png', img)[1].tobytes()
                    yield (b'--frame\r\n'b'Content-Type: image/png\r\n\r\n' + img + b'\r\n')
                except Exception as e:
                    print(str(e))
            else:
                blank_image = np.zeros((160, 160, 3), np.uint8)
                img = cv2.resize(blank_image, (0, 0), fx=0.5, fy=0.5)
                img = cv2.imencode('.png', img)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/png\r\n\r\n' + img + b'\r\n')


def tracking_gen():
    global prev
    global response
    frame_rate = 5
    try:
        cap.isOpened()
    except AttributeError:
        activate_stream()
    while (cap.isOpened()):
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        if ret:
            try:
                _, jpeg = cv2.imencode('.jpg', img)
                payload = {'image': jpeg.tobytes()}
                response = requests.post(ENDPOINT_tracking,
                                         files=payload).json()
            except Exception as e:
                print(str(e))
            img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
            frame = cv2.imencode('.png', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')
        else:
            break


@app.route('/streaming/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/streaming/recognition_feed')
def recognition_feed():
    return Response(recognition_gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/streaming/identified_feed')
def identified_feed():
    return Response(getIdentified(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/streaming/unidentified_feed')
def unidentified_feed():
    return Response(getUnidentified(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/streaming/tracked_feed')
def tracked_feed():
    return Response(tracking_gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, threaded=True)