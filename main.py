from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import cv2
import imutils
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to speak the recognized sign
def speak_sign(sign):
    engine.say(sign)
    engine.runAndWait()
    time.sleep(4) # 4 second delay

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
          "X", "Y", "Call Me", "Thank You", "Good Job", "I Love You"]


def detection_hand(frame):
    success, frame = cap.read()
    imgOutput = frame.copy()
    hands, img = detector.findHands(frame)
    # crop the image
    if hands:
        hand = hands[0]
        # bounding box
        x, y, w, h = hand['bbox']

        # white background
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            # center image in square
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)
            sign = labels[index]
            print(sign)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            # center image in square
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            sign = labels[index]
            print(sign)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50), (255, 0, 255),
                      cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        # Speak the recognized sign
        speak_sign(sign)

    return imgOutput

def visualizer():
    global cap
    ret, frame = cap.read()
    if ret == True:
        frame = imutils.resize(frame, width=640)
        frame = detection_hand(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=im)

        lblVideo.configure(image=img)
        lblVideo.image = img
        lblVideo.after(10, visualizer)
    else:
        lblVideo.image = ""
        lblInfoVideoPath.configure(text="")
        rad1.configure(state="active")
        rad2.configure(state="active")
        selected.set(0)
        btnEnd.configure(state="disabled")
        cap.release()


def reset():
    global cap
    if selected.get() ==1:
        path_video = filedialog.askopenfilename(filetypes=[
            ("all video format", ".mp4"),
            ("all video format", ".avi")])
        if len(path_video) > 0:
            btnEnd.configure(state="active")
            rad1.configure(state="disabled")
            rad2.configure(state="disabled")

            pathInputVideo = "..." + path_video[20:]
            lblInfoVideoPath.configure(text=pathInputVideo)
            cap = cv2.VideoCapture(path_video)
            visualizer()
    if selected.get() == 2:
        btnEnd.configure(state="active")
        rad1.configure(state="disabled")
        rad2.configure(state="disabled")
        lblInfoVideoPath.configure(text="")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        visualizer()


def cleanup():
    lblVideo.image = ""
    lblInfoVideoPath.configure(text="")
    rad1.configure(state="active")
    rad2.configure(state="active")
    selected.set(0)
    cap.release()


cap = None
root = Tk()

# Title label
lblInfo1 = Label(root, text="ASL Sign Language Detection", font="bold")
lblInfo1.grid(column=0, row=0, columnspan=2)

# Radio buttons
selected = IntVar()
rad1 = Radiobutton(root, text="Choose Video", width=20, value=1, variable=selected, command=reset)
rad2 = Radiobutton(root, text="Live Video", width=20, value=2, variable=selected, command=reset)
rad1.grid(column=0, row=1)
rad2.grid(column=1, row=1)

lblInfoVideoPath = Label(root, text="", width=20)
lblInfoVideoPath.grid(column=0, row=2)

lblVideo = Label(root)
lblVideo.grid(column=0, row=3, columnspan=2)

btnEnd = Button(root, text="Reset", state="disabled", command=cleanup)
btnEnd.grid(column=0, row=4, columnspan=2, pady=10)

root.mainloop()
