import cv2
import face_recognition
import os
import numpy as np

os.makedirs('faces', exist_ok=True)
name = input("Enter your name: ")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture Frame")
        break
    cv2.imshow('Face Recognition Attendance', frame)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        img_path = f'faces/{name}.jpg'
        cv2.imwrite(img_path, frame)
        print("Image saved at: ", img_path)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encoding = face_recognition.face_encodings(img_rgb)

        if encoding:
            np.save(f'faces/{name}_encodings.npy', encoding[0])
            print(f'Encoding saved for {name}')
        else:
            print("No Face Detected. Try again.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

