import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from keras import layers,losses
from keras import Sequential
import matplotlib.pyplot as plt
train_datagen=ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    'Dataset',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)
sh = train_generator[0][0].shape
def CNN():
    model = Sequential([
        layers.Conv2D(128,(3,3),padding='same',activation='relu',input_shape=(150,150,3)),
        layers.BatchNormalization(),
        layers.MaxPool2D((2,2)),
        layers.Conv2D(64,(3,3),padding='same',activation='relu',input_shape=(75,75,128)),
        layers.MaxPool2D((2,2)),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(37, 37, 64)),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(19, 19, 32)),
        layers.MaxPool2D((2, 2)),
        layers.TimeDistributed(layers.Flatten(input_shape=(sh[1:]))),
        layers.SimpleRNN(128, return_sequences=True),
        # layers.SimpleRNN(64, return_sequences=True),
        layers.SimpleRNN(64),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(64,activation='relu'),
        layers.Dense(32,activation='relu'),
        layers.Dense(4,activation='softmax')
    ])
    return model

def plot_history(history):
    # Plot loss
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show(block=True)

    # Plot accuracy
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show(block=True)
model = CNN()
# model.compile(optimizer='adam',loss="categorical_crossentropy",metrics=['accuracy',tf.keras.metrics.AUC()])
# history = model.fit(train_generator,epochs=10,validation_data=train_generator)
# model.save('CNN_RNN_save.h5',overwrite=True)
classLabels = ['Mild','Moderate','Non - demented','Very mild']
loaded_model = tf.keras.models.load_model('CNN_RNN_save.h5')
pic = cv2.imread('Dataset/Non_Demented/non_3.jpg')
pic = cv2.resize(pic,(150,150))
pic = pic.astype(np.float32)/255.0
pic = np.expand_dims(pic,axis=0)
x = loaded_model.predict(pic)
print(classLabels[x.argmax()])
