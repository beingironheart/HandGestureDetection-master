import pickle
import  cv2
import os
import tensorflow as tf
import tensorflow.keras.models
os.environ['CUDA_VISIBLE_DEVICES'] = ''
actions = ['hello', 'you']
filename = 'save_model.sav'
model = pickle.load(open(filename, 'rb'))
# print(model)

cap = cv2.VideoCapture(0)
while True:

    # Read feed
    ret, frame = cap.read()

    cv2.imshow('OpenCV Feed', frame)
    resized_arr = cv2.resize(frame, (200, 200))
    # break
    print(frame)
    print(resized_arr)
    break
    x_train = []
    y_train = []

    for feature, label in resized_arr:
        x_train.append(feature)
        y_train.append(label)
        print(x_train)
        # print(y_train)

    # Break gracefully
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
