import cv2
import face_recognition
import os
import numpy as np


def load_encodings(encodings_path):
    encodings = []
    class_names = []
    for file in os.listdir(encodings_path):
        if file.endswith("_encodings.npy"):
            class_name = file.split("_")[0]
            encoding = np.load(os.path.join(encodings_path, file))
            encodings.append(encoding)
            class_names.append(class_name)
    return encodings, class_names
encodings_path = "faces"
known_encodings, class_names = load_encodings(encodings_path)
print(f"Loaded classes: {class_names}")

cap = cv2.VideoCapture(0)
scale = 0.25
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encoding, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = class_names[first_match_index].upper()

        top, right, bottom, left = [int(val/scale) for val in face_location]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom-20), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Face Recognition - Press 'q' to quit.", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()