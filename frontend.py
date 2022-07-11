import io
import json
import os
import cvzone
import torch
from time import time, sleep
import time

import imutils
import requests

from flask import Flask, render_template, Response, request, jsonify

import cv2
import numpy as np

import postprocessing
from trackers import KFTracker

ENDPOINT_recognition = 'http://127.0.0.1:8000/api/facial_recognition/inference/'
ENDPOINT_tracking = 'http://127.0.0.1:8000/api/people_detection/inference/'

cap = None
cap_surveillance = None
video = None
currentframe = None
response = []

model = None

identified = []
unidentified = []
prev = 0
prevIdent = 0
prevUnident = 0
numPeople = 0;
numObjects = 0

def activate_stream():
    global cap
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'vp80')

    global currentframe
    currentframe = 0

    global prev
    prev = 0

def activate_surveillance():
    global numPeople
    global numObjects
    numPeople = 0
    numObjects = 0
    global prev
    prev = 0
    global cap_surveillance
    RTSP_URL = 'rtsp://admin:Bo0214925184!21@105.233.39.146:5544/Streaming/Channels/101'
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
    cap_surveillance = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap_surveillance.set(3, 680)
    cap_surveillance.set(4, 480)
    fourcc = cv2.VideoWriter_fourcc(*'vp80')

    global currentframe
    currentframe = 0

    global model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def deactivate_stream():
    global cap
    cap = None
    global video
    video = None
    global currentframe
    currentframe = None

def deactivate_stream_surveillance():
    global cap_surveillance
    cap_surveillance = None
    global video
    video = None
    global currentframe
    currentframe = None

# Initialise Flask
app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    deactivate_stream()
    deactivate_stream_surveillance()
    return render_template('index.html')

@app.route('/stream')
def stream():
    deactivate_stream()
    deactivate_stream_surveillance()
    if not cap_surveillance:
        activate_surveillance()
    return render_template('stream.html')

@app.route('/recognition')
def recognition():
    deactivate_stream()
    deactivate_stream_surveillance()
    if not cap == None:
        activate_stream()
    return render_template('recognition.html')

@app.route('/tracking')
def tracking():
    deactivate_stream()
    deactivate_stream_surveillance()
    if not cap == None:
        activate_stream()
    return render_template('tracking.html')

@app.route('/detection')
def detection():
    deactivate_stream()
    deactivate_stream_surveillance()
    if not cap == None:
        deactivate_stream_surveillance()
    return render_template('detection.html')

@app.route('/return')
def main_site():
    deactivate_stream()
    deactivate_stream_surveillance()
    main_url = request.host_url.replace("8001","8000").replace('return','/')
    return render_template('return.html', main_url=main_url)

def gen():
    try:
        cap.isOpened()
    except AttributeError:
        activate_stream()
    while cap.isOpened():
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        if ret:
            img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
            frame = cv2.imencode('.png', img)[1].tobytes()
            yield b'--frame\r\n'b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n'
        else:
            break

def gen_surveillance():
    global model
    global prev
    global response
    tracker = KFTracker(1 / 1., 0, file_name=None)
    try:
        cap_surveillance.isOpened()
    except AttributeError:
        activate_surveillance()
    while cap_surveillance.isOpened():
        ret, img = cap_surveillance.read()
        if ret:
            try:
                _, jpeg = cv2.imencode('.jpg', img)
                payload = {'image': jpeg.tobytes()}
                response = requests.post(ENDPOINT_tracking,
                                         files=payload).json()
                people = response.get('People')
                img = postprocessing.drawPeople(img, people)
                centroids = []
                for person in people:
                    centroids.append(person.get('centroid'))
                tracks = postprocessing.getTracks(tracker, centroids)
                img = postprocessing.drawTracks(img, tracks['Tracks'])
                img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
                frame = cv2.imencode('.png', img)[1].tobytes()
                yield b'--frame\r\n'b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n'
            except Exception as e:
                print(str(e))
        else:
            break

def object_detection():
    global model
    global response
    global numObjects
    global numPeople
    try:
        cap_surveillance.isOpened()
    except AttributeError:
        activate_surveillance()
    while cap_surveillance.isOpened():
        ret, img = cap_surveillance.read()
        if ret:
            try:
                _, jpeg = cv2.imencode('.jpg', img)
                payload = {'image': jpeg.tobytes()}
                response = requests.post(ENDPOINT_tracking,
                                         files=payload).json()
                people = response.get('People')
                numPeople = len(people)
                img = postprocessing.drawPeople(img, people)
                objects = response.get('Objects')
                numObjects = len(objects)
                img = postprocessing.drawObjects(img, objects)
                img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
                frame = cv2.imencode('.png', img)[1].tobytes()
                yield b'--frame\r\n'b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n'
            except Exception as e:
                print(str(e))
        else:
            break

def getPeople():
    global numPeople
    try:
        return numPeople
    except:
        return 0

def getObjects():
    global numObjects
    try:
        return numObjects
    except:
        return 0


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
    while cap.isOpened():
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
            yield b'--frame\r\n'b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n'
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
    while cap.isOpened():
        time_elapsed = time.time() - prevIdent
        if time_elapsed > 1. / frame_rate:
            prevIdent = time.time()
            if len(identified) > 0:
                try:
                    img = postprocessing.gen_faces(identified)
                    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                    img = cv2.imencode('.png', img)[1].tobytes()
                    yield b'--frame\r\n'b'Content-Type: image/png\r\n\r\n' + img + b'\r\n'
                except Exception as e:
                    print(str(e))
            else:
                blank_image = np.zeros((160, 160, 3), np.uint8)
                img = cv2.resize(blank_image, (0, 0), fx=0.5, fy=0.5)
                img = cv2.imencode('.png', img)[1].tobytes()
                yield b'--frame\r\n'b'Content-Type: image/png\r\n\r\n' + img + b'\r\n'

def getUnidentified():
    global unidentified
    global prevUnident
    frame_rate = 0.5
    try:
        cap.isOpened()
    except AttributeError:
        activate_stream()
    while cap.isOpened():
        time_elapsed = time.time() - prevIdent
        if time_elapsed > 1. / frame_rate:
            prevIdent = time.time()
            if len(unidentified) > 0:
                try:
                    img = postprocessing.gen_faces(unidentified)
                    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                    img = cv2.imencode('.png', img)[1].tobytes()
                    yield b'--frame\r\n'b'Content-Type: image/png\r\n\r\n' + img + b'\r\n'
                except Exception as e:
                    print(str(e))
            else:
                blank_image = np.zeros((160, 160, 3), np.uint8)
                img = cv2.resize(blank_image, (0, 0), fx=0.5, fy=0.5)
                img = cv2.imencode('.png', img)[1].tobytes()
                yield b'--frame\r\n'b'Content-Type: image/png\r\n\r\n' + img + b'\r\n'


def tracking_gen():
    global prev
    global response
    frame_rate = 5
    try:
        cap.isOpened()
    except AttributeError:
        activate_stream()
        tracker = KFTracker(1 / 1., 0, file_name=None)
    while cap.isOpened():
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        if ret:
            try:
                _, jpeg = cv2.imencode('.jpg', img)
                payload = {'image': jpeg.tobytes()}
                response = requests.post(ENDPOINT_tracking,
                                         files=payload).json()
                people = response.get('People')
                img = postprocessing.drawPeople(img, people)
                centroids = []
                for person in people:
                    centroids.append(person.get('centroid'))
                tracks = postprocessing.getTracks(tracker, centroids)
                img = postprocessing.drawTracks(img, tracks['Tracks'])
            except Exception as e:
                print(str(e))
            img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
            frame = cv2.imencode('.png', img)[1].tobytes()
            yield b'--frame\r\n'b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n'
        else:
            break


@app.route('/streaming/video_feed')
def video_feed():
    return Response(gen_surveillance(),
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

@app.route('/streaming/detection_feed')
def detection_feed():
    return Response(object_detection(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/streaming/detection_feed/counts')
def detection_feed_count():
    return jsonify({
        "people": getPeople(),
        "objects": getObjects()
    })


if __name__ == '__main__':
    app.run(debug=True, threaded=True, host="0.0.0.0")