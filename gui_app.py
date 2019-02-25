import Tkinter as tk
import urllib
import cv2
import numpy as np
import keras
import tkMessageBox as tb

top = tk.Tk()
# Replace the URL with your own IPwebcam shot.jpg IP:port

url = 'http://192.168.43.1:8080/shot.jpg'
model_filename = '/home/techvamp/Documents/Project/chillly_work/code/model/model_v1_finetuned.h5'

model = keras.models.load_model(model_filename)



image = 0
def open_cam():

    while True:

        # Use urllib to get the image from the IP camera
        imgResp = urllib.urlopen(url)

        # Numpy to convert into a array
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)

        print imgNp

        # Finally decode the array to OpenCV usable format ;)
        img = cv2.imdecode(imgNp, -1)

        # put the image on screen
        cv2.imshow('IPWebcam', img)

        # To give the processor some less stress
        # time.sleep(0.1)

        # Quit if q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def capture():
    while True:

        imgResp = urllib.urlopen(url)


        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)



    # Finally decode the array to OpenCV usable format ;)
        img = cv2.imdecode(imgNp, -1)


    # save the image to loacal
        cv2.imwrite('input.jpg', img)

        print imgNp

        return imgNp

        break

def get_image():

    camera = capture()

    print camera

    #imgnp = np.array(bytearray(image.read()), dtype=np.uint8)
    #img = cv2.imdecode(imgnp, -1)
    #cv2.imshow('sample', img)
    image_name = '/home/techvamp/PycharmProjects/sample/input.jpg'

    image = cv2.imread(image_name)
    image = cv2.resize(image, (150, 150))
    image = np.array(image, np.float32) / 255.0
    input_tenser = np.expand_dims(image, axis=0)

    inference_result = model.predict(input_tenser)[0]
    if (inference_result[0] > inference_result[1]):
        tb.showinfo('chilly', 'It is s a chilly')
    else:
         tb.showinfo('capsicum', 'it is capsicum')


button = tk.Button(top, text = 'open cam', command = open_cam)
captureb = tk.Button(top, text = 'capture', command = capture)
show = tk.Button(top, text = 'show', command = get_image)


button.pack()
captureb.pack()
show.pack()
top.mainloop()