import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
import tempfile
import time

distraction_model = tf.keras.models.load_model("./model/Distraction.h5")
drowsiness_model = tf.keras.models.load_model("./model/Drowsiness.h5")
face = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
leye = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
font = cv2.FONT_HERSHEY_PLAIN


def render_ddd():
    st.image("./Distraction.png", width=720)

    # Image classification
    image_file = st.file_uploader(
        "Upload an image", type=['jpeg', 'png', 'jpg'])

    if image_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        if st.button("Process image"):
            # Show image
            st.image(img, channels="BGR", width=480)
            label = detect_distraction(img)
            if label == "Safe driving":
                st.success(label)
            else:
                st.error(label)

    # Video classification
    st.subheader("OR")

    video_file = st.file_uploader("Upload a video", type=['mp4', 'wav'])

    stframe = st.image([])
    label_placeholder = st.empty()

    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=True)
        tfile.write(video_file.read())

        if st.button("Process video"):

            score = 0

            cap = cv2.VideoCapture(tfile.name)
            cap.set(cv2.CAP_PROP_FPS, 25)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

            while (cap.isOpened()):
                ret, frame = cap.read()
                if not ret:
                    break

                label = detect_distraction(frame)

                if label == "Safe driving":
                    label_placeholder.success(label)
                else:
                    label_placeholder.error(label)

                frm = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                stframe.image(frm, width=480)

            cap.release()

    # Webcam classification
    st.subheader("OR")
    run = st.checkbox('Turn on your webcam')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FPS, 25)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    label_placeholder_webcam = st.empty()

    while run:
        _, frame = camera.read()
        label = detect_distraction(frame)
        if label == "Safe driving":
            label_placeholder_webcam.success(label)
        else:
            label_placeholder_webcam.error(label)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame, width=480)
    else:
        camera.release()


def render_drowsiness():
    st.image("./Drowsiness.jpeg", width=720)

    # Image classification
    image_file = st.file_uploader(
        "Upload an image", type=['jpeg', 'png', 'jpg'])

    if image_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        if st.button("Process image"):
            # Show image
            st.image(img, channels="BGR", width=480)
            label = detect_drowsiness(img)
            if label == "Open eyes":
                st.success(label)
            else:
                st.error(label)

    # Video classification
    st.subheader("OR")

    video_file = st.file_uploader("Upload a video", type=['mp4'])

    stframe = st.image([])
    label_placerholder = st.empty()
    alert_video = st.empty()

    if video_file is not None:

        tfile = tempfile.NamedTemporaryFile(delete=True)
        tfile.write(video_file.read())

        if st.button("Process video"):

            score = 0

            cap = cv2.VideoCapture(tfile.name)
            cap.set(cv2.CAP_PROP_FPS, 25)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

            while (cap.isOpened()):
                ret, frame = cap.read()
                if not ret:
                    break

                label = detect_drowsiness(frame)

                if label == "Closed eyes":
                    label_placerholder.error(label)
                    score += 1
                else:
                    label_placerholder.success(label)
                    score -= 1

                frm = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if score < 0:
                    score = 0
                if score > 5:
                    alert_video.error("The driver is sleepy. STOP DRIVING NOW")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                stframe.image(frm, width=480)

            cap.release()

    # Webcam classification
    st.subheader("OR")
    run = st.checkbox('Turn on your webcam')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FPS, 25)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    label_placeholder_webcam = st.empty()
    alert_webcam = st.empty()

    score = 0

    while run:    

        _, frame = camera.read()

        label = detect_drowsiness(frame)

        if label == "Closed eyes":
            label_placeholder_webcam.error(label)
            score += 1
        else:
            label_placeholder_webcam.success(label)
            score -= 1

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if score < 0:
            score = 0
        if score > 2:
            alert_webcam.error("The driver is sleepy. STOP DRIVING NOW")
        
        FRAME_WINDOW.image(frame, width=480)
    else:
        camera.release()

def detect_distraction(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    img = np.expand_dims(img, axis=0)
    img = img / 255.
    pred_class = distraction_model.predict_classes(img)[0]
    if pred_class == 0: return "Safe driving"
    if pred_class == 1: return "Texting - right"
    if pred_class == 2: return "Talking on the phone - right"
    if pred_class == 3: return "Texting - left"
    if pred_class == 4: return "Talking on the phone - left"
    if pred_class == 5: return "Operating the radio"
    if pred_class == 6: return "Drinking"
    if pred_class == 7: return "Reaching behind"
    if pred_class == 8: return "Hair and makeup"
    if pred_class == 9: return "Talking to passenger"
    return "?"

def detect_drowsiness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    label = ''

    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    for (x, y, w, h) in right_eye:
        r_eye = img[y:y+h, x:x+w]
        r_eye = cv2.resize(r_eye, (224, 224))
        r_eye = np.expand_dims(r_eye, axis=0)
        r_eye = r_eye / 255.
        r_pred = drowsiness_model.predict(r_eye)
        if r_pred[0][0] > 0.5:
            label = "Open eyes"
        else:
            label = "Closed eyes"
        break

    for (x, y, w, h) in left_eye:
        l_eye = img[y:y+h, x:x+w]
        l_eye = cv2.resize(l_eye, (224, 224))
        l_eye = np.expand_dims(l_eye, axis=0)
        l_eye = l_eye / 255.
        r_pred = drowsiness_model.predict(l_eye)
        if r_pred[0][0] > 0.5:
            label = "Open eyes"
        else:
            label = "Closed eyes"
        break

    return label


def main():
    st.title("Driver Monitor App :car:")
    st.write("Choose the model from the sidebar")

    model = ["Driver Distraction Detection", "Driver Drowsiness Detection"]

    choice = st.sidebar.selectbox("Select model", model)

    if choice == "Driver Distraction Detection":
        render_ddd()
    elif choice == "Driver Drowsiness Detection":
        render_drowsiness()

    st.sidebar.title("About us")
    st.sidebar.info("""
        Trung Duc Nguyen - 13826211 \n
        Hailey Nguyen - 13665132 \n
        Prakrit Sethi - 13669130 \n
        """)

if __name__ == "__main__":
    main()
