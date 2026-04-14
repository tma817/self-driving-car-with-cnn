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
app = Flask(__name__) #__main__
maxSpeed = 10

def preProcessing(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255

    return img


@sio.on('telemetry')
def telemetry(sid, data):
    global prev_steering
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = preProcessing(image)
    image = np.array([image])

    steering = float(model.predict(image))
    steering = steering * 1.1  
    steering = 0.7 * prev_steering + 0.3 * steering
    steering = max(min(steering, 1), -1)
    prev_steering = steering

    if abs(steering) > 0.2:
        maxSpeed = 2
    elif abs(steering) > 0.1:
        maxSpeed = 3
    elif abs(steering) > 0.05:
        maxSpeed = 4
    else:
        maxSpeed = 4.5

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
        'steering_angle' : steering.__str__(),
        'throttle' : throttle.__str__()
    })

if __name__ == "__main__":
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)