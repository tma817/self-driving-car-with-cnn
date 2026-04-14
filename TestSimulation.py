import os
print('Setting Up ...')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import socketio
import eventlet
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask
import base64
from io import BytesIO
from PIL import Image
import cv2

prev_steering = 0.0
sio = socketio.Server()
app = Flask(__name__)
maxSpeed = 10

def preProcessing(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img

@sio.on('telemetry')
def telemetry(sid, data):
    global prev_steering

    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = preProcessing(image)
    image = np.array([image])

    # Raw steering angle value from the model
    steering = float(model.predict(image, verbose=0))

    # --- Modification to steering angle ---
    steering = steering * 1.05
    steering = 0.75 * prev_steering + 0.25 * steering
    steering = max(min(steering, 0.7), -0.7)
    prev_steering = steering

    if abs(steering) > 0.30:
        maxSpeed = 1.8
    elif abs(steering) > 0.18:
        maxSpeed = 2.5
    elif abs(steering) > 0.08:
        maxSpeed = 3.5
    else:
        maxSpeed = 4.5

    # --- Modify throttle based on current speed
    throttle = 1.0 - speed / maxSpeed
    if speed > maxSpeed:
        throttle = -0.3

    print(f'{throttle}, {steering}, {speed}')
    sendControl(steering, throttle)

@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    sendControl(0, 0)

def sendControl(steering, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering),
        'throttle': str(throttle)
    })

if __name__ == "__main__":
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)