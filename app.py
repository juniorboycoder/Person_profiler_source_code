# importing libraries
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
import cvlib as cv
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
import numpy as np

try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

import numpy as np




model = tf.keras.models.load_model("complexion4.model")
### load file



genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
#ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

#faceNet=cv2.dnn.readNet(faceModel,faceProto)
#ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)


# Here is the function for UI
def main():
    st.title("Person-Profiler(complexion and gender) application(face detection and live detection) ")
   
    st.write("--Use operations in the side bar")

    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("**Select an option ** ")
    st.sidebar.write("")

    activities = [
                  "Picture(Face detection)", "Camera(live detection)"]
    choice = st.sidebar.selectbox("select an option", activities)

   

    if choice == "Picture(Face detection)":
        image_file = st.file_uploader(
            "Upload image", type=['jpeg', 'png', 'jpg', 'webp'])

        if image_file:

            image = Image.open(image_file)

            if st.button("Process"):

                # result_img is the image with rectangle drawn on it (in case there are faces detected)
                # result_faces is the array with co-ordinates of bounding box(es)
                image = np.array(image.convert('RGB'))

                classes = ["black", "white","yellow"]

            
                #run = st.checkbox('Run')
                FRAME_WINDOW = st.image([])
                
                #webcam = cv2.VideoCapture(0)
               

                # read frame from webcam 
                #status, frame = webcam.read()

                # apply face detection
                face, confidence = cv.detect_face(image)


                # loop through detected faces
                for idx, f in enumerate(face):

                    # get corner points of face rectangle        
                    (startX, startY) = f[0], f[1]
                    (endX, endY) = f[2], f[3]

                    # draw rectangle over face
                    cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)

                    # crop the detected face region
                    face_crop = np.copy(image[startY:endY,startX:endX])

                    if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                        continue

                    # preprocessing for gender detection model
                    face_crop = cv2.resize(face_crop, (224,224))
                    face_crop2 = cv2.resize(face_crop, (227,227))
                    face_crop = face_crop.astype("float") / 255.0
                    face_crop = img_to_array(face_crop)
                    face_crop = np.expand_dims(face_crop, axis=0)

                    # apply gender detection on face
                    conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

                    # get label with max accuracy
                    idx = np.argmax(conf)
                    label = classes[idx]

                    label = "{}: {:.2f}%".format(label, conf[idx] * 100)

                    Y = startY - 10 if startY - 10 > 10 else startY + 10
                    Y2 = startY - 25 if startY - 15 > 10 else startY + 25
                    #gender
                    blob=cv2.dnn.blobFromImage(face_crop2, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                    genderNet.setInput(blob)
                    genderPreds=genderNet.forward()
                    gender=genderList[genderPreds[0].argmax()]
                    #print(f'Gender: {gender}')

                    # write label and confidence above face rectangle
                    cv2.putText(image, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)
                    cv2.putText(image, f'{gender}', (startX, Y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
                    FRAME_WINDOW.image(image)


    if choice == "Camera(live detection)":
        
        classes = ["black", "white","yellow"]
        st.header("Webcam Live Feed")
        run = st.checkbox('Run')
        FRAME_WINDOW = st.image([])
        
        webcam = cv2.VideoCapture(0)
        while run:
            

             # read frame from webcam 
            status, frame = webcam.read()

            # apply face detection
            face, confidence = cv.detect_face(frame)


            # loop through detected faces
            for idx, f in enumerate(face):

                # get corner points of face rectangle        
                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]

                # draw rectangle over face
                cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

                # crop the detected face region
                face_crop = np.copy(frame[startY:endY,startX:endX])

                if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                    continue

                # preprocessing for gender detection model
                face_crop = cv2.resize(face_crop, (224,224))
                face_crop2 = cv2.resize(face_crop, (227,227))
                face_crop = face_crop.astype("float") / 255.0
                face_crop = img_to_array(face_crop)
                face_crop = np.expand_dims(face_crop, axis=0)

                # apply gender detection on face
                conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

                # get label with max accuracy
                idx = np.argmax(conf)
                label = classes[idx]

                label = "{}: {:.2f}%".format(label, conf[idx] * 100)

                Y = startY - 10 if startY - 10 > 10 else startY + 10
                Y2 = startY - 25 if startY - 15 > 10 else startY + 25
        #gender
                blob=cv2.dnn.blobFromImage(face_crop2, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds=genderNet.forward()
                gender=genderList[genderPreds[0].argmax()]
                #print(f'Gender: {gender}')

                # write label and confidence above face rectangle
                cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'{gender}', (startX, Y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
                FRAME_WINDOW.image(frame)
        else:
            st.write('Stopped')


if __name__ == "__main__":
    main()
