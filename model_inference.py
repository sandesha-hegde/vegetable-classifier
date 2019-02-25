import keras
import numpy as np
import cv2
import time

image_test = '/home/techvamp/PycharmProjects/sample/input.jpg'
image_filename_A = '/home/techvamp/Documents/Project/A/IMG_20180415_201441.jpg'
image_filename = '/home/techvamp/Documents/Project/Capsicum/IMG_20180708_152841.jpg'
model_filename = '/home/techvamp/Documents/Project/chillly_work/code/model/model_v1_finetuned.h5'

t1 = time.time()

model = keras.models.load_model(model_filename)

t2 = time.time()

tdif = t2 - t1
print(str('time spent '+str(tdif)))

t1 = time.time()
image = cv2.imread(image_test)
image = cv2.resize(image, (150, 150))
image = np.array(image, np.float32) / 255.0


input_tenser = np.expand_dims(image, axis=0)
inference_result = model.predict(input_tenser)[0]

t2 = time.time()

tdif = t2 - t1
print(str('time spent '+str(tdif)))

if (inference_result[0] > inference_result[1]):
    print "This is a chilly image"
else:
    print "This is a capsicum image"

print "working..."