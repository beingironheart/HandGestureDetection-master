from flask import Flask, render_template, Response
from flask import jsonify

import threading
import argparse

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

# Global variable to store latest result.
glob = 'Initializing...'
feed_image = ''

lock = threading.Lock()

mp_holistic = mp.solutions.holistic  # holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing Utilities

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    # Get real time prediction
    return jsonify(
        data=glob
    )

def mediapipe_detection(image, model):
    # Color Conversion BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False  # image is no longer writeable
    results = model.process(image)  # make prediction
    image.flags.writeable = True  # Image is now writable
    # Color Conversion RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks,
                              mp_holistic.FACEMESH_CONTOURS)  # draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks,
                              mp_holistic.POSE_CONNECTIONS)  # draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # draw left_hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # draw right_hand connections


def draw_styled_landmarks(image, results):
    # draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(
                                  color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(
                                  color=(80, 256, 121), thickness=1, circle_radius=1)
                              )
    # draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    # draw left_hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    # draw right_hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten(
    ) if results.pose_landmarks else np.zeros(33*4)

    face = np.array([[res.x, res.y, res.z]

    for res in results.face_landmarks.landmark]).flatten(
    ) if results.pose_landmarks else np.zero(468 * 3)
    
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
    ) if results.left_hand_landmarks else np.zeros(21*3)
    
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
    ) if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, face, left_hand, right_hand])

def extract_keypoints(results):
    key1 = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    key2 = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([key1, key2, lh, rh])

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Action that we try to detect
# actions = np.array(['Hello', 'Thanks', 'I love you'])
actions = np.array(['bye', 'call me please', 'good', 'hello'])
# actions = np.array(['hello', 'bye', 'you'])
# actions = np.array(['bye', 'come together', 'Good Morning', 'hello', 'high five', 'how are you?', 'idle', 'I love you', 'listen up', 'Looser', 'me', 'namaste', 'Not Okay', 'okay', 'peace', 'Please call me..!', 'Rock', 'sorry', 'stop it..!', 'unique', 'wrong', 'you'])

# 30 videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

label_map = {label: num for num, label in enumerate(actions)}
# print(label_map)

sequences, labels = [], []
'''
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(
                sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

x = np.array(sequences)
# print(x.shape)
y = to_categorical(labels).astype(int)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True,
          activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# res = [.7, 0.2, 0.1]
# print(actions[np.argmax(res)])

model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

# training model with sequential nuaral network

# model.fit(x_train, y_train, epochs=2000, callbacks=[tb_callback])

# model.summary()

res = model.predict(x_test)
# print(np.sum(res[0]))
print(actions[np.argmax(res[0])])

actions[np.argmax(y_test[0])]   

# model.save('action.h5')

from keras.models import load_model

model = load_model('action.h5')

model.summary()

# model.load_weights('action.h5')

yhat = model.predict(x_train)

ytrue = np.argmax(y_train, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(ytrue)
print(yhat)

mcm = multilabel_confusion_matrix(ytrue, yhat)
print(mcm)
a = accuracy_score(ytrue, yhat)
print(a)    
'''

from keras.models import load_model

# model = load_model('trained_model_3.h5')
model = load_model('trained_model_4.h5')
model.summary()

cap = None

import json
def classify():
    sequence = []
    sentence = []
    threshold = 0.4
    global cap 
    cap = cv2.VideoCapture(0)

    cap.set(3,600)
    cap.set(4,400)

    # set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()  # reading the video frames from camera

            # make detaction
            image, results = mediapipe_detection(frame, holistic)

            # draw landmarks
            draw_styled_landmarks(image, results)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # result_test = extract_keypoints(results)
            # np.save('0', result_test)
            # prediction logic
            keypoints = extract_keypoints(results)
            sequence.insert(0, keypoints)
            sequence = sequence[:30]

            if len(sequence) == 30:
                # pred_array_scaled = np.expand_dims(pred_array_scaled, axis=0)
                # res = model.predict(np.expand_dims(sequence, axis=0))[0]
                # pred_array = cv2.imread(frame)
                # pred_array = np.array(pred_array)
                # pred_array_resized = cv2.resize(pred_array,(224,224,))
                # pred_array_scaled = np.array(pred_array_resized)/255
                # res = model.predict(np.expand_dims(sequence, axis=0)) # orig
                _frame = cv2.resize(frame, (150, 150)).astype("float32")
                res = model.predict(np.expand_dims(_frame, axis=0))[0]
                # print(res)
                # print(actions[np.argmax(res)])
                global glob
                glob = actions[np.argmax(res)]
                global feed_image
                feed_image = image
            # # show to screen
            winname = "OpenCV Feed"
            cv2.namedWindow(winname)
            cv2.moveWindow(winname, 20, 310)
            cv2.imshow(winname, image)

            # braking the video loop using 'q'
            if cv2.waitKey(10) & 0xff == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

@app.route('/refresh_data')
def refresh_data():
    return jsonify({'data': glob})

@app.route('/refresh_image')
def refresh_image():
    # return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return jsonify({'data': feed_image})

def generate_frames():
    while True:
        success, frame_for_feed = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame_for_feed)
            frame_for_feed=buffer.tobytes()
        yield(b'--frame\r\n', b'Content-Type: image/jpeg\r\n\r\n', frame_for_feed, b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument("-i", "--ip", type=str, required=True,
  help="ip address of the device")
  ap.add_argument("-o", "--port", type=int, required=True,
  help="ephemeral port number of the server (1024 to 65535)")
  args = vars(ap.parse_args())
  t = threading.Thread(target=classify)
  t.daemon = True
  t.start()
  app.run(host=args["ip"],port=args["port"],debug=True,threaded=True,use_reloader=False)
