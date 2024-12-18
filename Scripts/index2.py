import tensorflow as tf
import cv2
import numpy as np

classLabels = ['Mild','Moderate','Non - demented','Very mild']
loaded_model = tf.keras.models.load_model('CNN_RNN_save.h5')
pic = cv2.imread('Dataset/Non_Demented/non_3.jpg')
pic = cv2.resize(pic,(150,150))
pic = pic.astype(np.float32)/255.0
pic = np.expand_dims(pic,axis=0)
x = loaded_model.predict(pic)
print(classLabels[x.argmax()])