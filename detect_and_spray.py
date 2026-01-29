import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("model/cnn_model.h5")
labels = ["Healthy", "Diseased"]

img = cv2.imread("test_leaf.jpg")
img = cv2.resize(img, (128,128))
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
result = labels[np.argmax(prediction)]

print("Detected:", result)

if result == "Diseased":
    print("Sprayer Activated ")
else:
    print("No spraying required ")
