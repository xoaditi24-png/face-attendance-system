import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime

KNOWN_DIR = "known_faces"
CSV_FILE = "attendance.csv"

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

def train():
    faces, labels, names = [], [], []
    label_id = 0
    for person in os.listdir(KNOWN_DIR):
        person_dir = os.path.join(KNOWN_DIR, person)
        if not os.path.isdir(person_dir):
            continue
        names.append(person)
        for img_file in os.listdir(person_dir):
            img = cv2.imread(os.path.join(person_dir, img_file), cv2.IMREAD_GRAYSCALE)
            faces.append(img)
            labels.append(label_id)
        label_id += 1
    recognizer.train(faces, np.array(labels))
    return names

def mark_attendance(name):
    if not os.path.exists(CSV_FILE):
        pd.DataFrame(columns=["Name", "Time"]).to_csv(CSV_FILE, index=False)
    df = pd.read_csv(CSV_FILE)
    today = datetime.now().strftime("%Y-%m-%d")
    if not ((df["Name"] == name) & (df["Time"].str.startswith(today))).any():
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df.loc[len(df)] = [name, now]
        df.to_csv(CSV_FILE, index=False)
        print(f"Marked attendance: {name} at {now}")

names = train()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_img)
        if confidence < 90:
            name = names[label]
            mark_attendance(name)
        else:
            name = "Unknown"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
