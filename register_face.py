import cv2
import os

name = input("Enter person's name: ").strip()
save_dir = os.path.join("known_faces", name)
os.makedirs(save_dir, exist_ok=True)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
count = 0

print("Look at the camera. Capturing 20 images... press ESC to stop early.")

while count < 20:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(save_dir, f"{count}.jpg"), face_img)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Registering Face", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Captured {count} images for {name}.")