from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = ''

train = ImageDataGenerator(rescale= 1/255)
validation = ImageDataGenerator(rescale=1/255)
train_dataset = train.flow_from_directory("/home/meeni/PycharmProjects/Facedetaction/Action_Frame/",
                                          target_size = (200, 200),
                                          batch_size= 3,
                                          class_mode='categorical')
validation_dataset = train.flow_from_directory("/home/meeni/PycharmProjects/Facedetaction/Action_validation/",
                                             target_size=(200, 200),
                                             batch_size=3,
                                             class_mode='categorical')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3),activation = 'relu',input_shape = (200,200,3)),
    tf.keras.layers.MaxPool2D(2,2),
    #
    tf.keras.layers.Conv2D(32, (3,3),activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    #
    tf.keras.layers.Conv2D(64, (3,3),activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    ##
    tf.keras.layers.Flatten(),
    ##
    tf.keras.layers.Dense(512,activation = 'relu'),
    ##
    tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics = ['accuracy'])

model_fit = model.fit(train_dataset,
                      epochs=100,
                      steps_per_epoch=10,
                      validation_data=validation_dataset)

filename = 'save_model.sav'
pickle.dump(model, open(filename, 'wb'))
