import cv2
import numpy as np
import os
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('Action_Frame')

# Actions that we try to detect
actions = np.array(['test2'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 30
for action in actions:
    try:
        os.makedirs(os.path.join(DATA_PATH, action))
    except:
        pass

cap = cv2.VideoCapture(0)
i = 300
# i = 450

for frame_num in range(0, 150):
# for frame_num in range(450, 601):

    # Read feed
    ret, frame = cap.read()

    cv2.imshow('OpenCV Feed', frame)

    path = os.path.join(DATA_PATH, action) + '/Frame'+str(i)+'.jpg'
    cv2.imwrite(path, frame)
    print(i)
    i += 1
    cv2.waitKey(100)
    print('Waiting... ')
    # Break gracefully
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
