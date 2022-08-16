# final version
from turtle import color
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# load the face detector cascade and smile detector CNN
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('./model/bestCNN_model.h5')
# model = load_model('model/model01.h5')
# model = load_model('model/model_cnn_lenet_early_stop3.h5')
# model = load_model('best_model.h5')
# model = load_model('trans4_model.h5')

# if a video path was not supplied, grab the reference to the webcam

camera = cv2.VideoCapture(1)

while True:
    # grab the current frame
    (grabbed, frame) = camera.read()
    # print(frame.shape)

    # reszie the frame, convert it to grayscale, then clone the
    # original frame so we can draw on it later in the program
    frame = imutils.resize(frame, width = 600)
    
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    frameClone = frame.copy()

    # detect faces in the input frame, then clone the frame so
    # that we can draw on it
    # grey
    rects = detector.detectMultiScale(frame, scaleFactor = 1.1, minNeighbors = 5,
        minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)

    for (fX, fY, fW, fH) in rects:
        # extract the ROI of the face from the grayscale image,
        # resize it to a fixed 28x28 pixels, and then prepare the
        # ROI for classification via CNN
        
        roi = frame[fY: fY + fH, fX: fX + fW]

        roi = cv2.resize(roi, (64, 64))
    

        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        print('roi', roi.shape)


        roi = np.expand_dims(roi, axis = 0)
        print('roi', roi.shape)


        ####


        # determine the probabilities of both "smiling" and "not similing"
        # then set the label accordingly
        (notSmiling, smiling) = model.predict(roi)[0]
        print(model.predict(roi)[0])
        label = "Smiling" if smiling > notSmiling else "Not Smiling"

        # display the label and bounding box rectangle on the output frame
        cv2.putText(frameClone, label, (fX, fY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
            (0, 0, 255), 2)

    # show our detected faces along with smiling/not smiling labels
    cv2.imshow("Face", frameClone)

    # if 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
