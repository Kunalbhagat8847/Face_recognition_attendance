import face_recognition
import os
import cv2
import numpy as np
import datetime
import csv
import time
import DateTime
# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Load known face images and encodings
known_face_encodings = []
known_faces_names = []

for image_file in ["tata.jpg","tesla.jpg"]:
    image = face_recognition.load_image_file(image_file)
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_faces_names.append(os.path.splitext(image_file)[0])

# Initialize student list
students = known_faces_names.copy()

# Wait for camera to initialize

time.sleep(2)  # Add a delay of 2 seconds

while True:
    # Read frame and resize
    ret, frame = video_capture.read()
    if not ret:
        break
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR to RGB

    # Find faces, encodings, and names
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = ""
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)
        if matches[best_match_index]:
            name = known_faces_names[best_match_index]

        face_names.append(name)

    # Mark present faces and update student list
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if name in students:
            students.remove(name)
            print(f"{name} is present.")

            # Draw rectangle and name with error handling
            try:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            except:
                print("Error drawing rectangle and name.")

    # Save attendance with error handling
    now = datetime.datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    file_name = f"{current_date}.csv"

    try:
        with open(file_name, 'a', newline='') as f:
            writer = csv.writer(f)
            for name in students:
                current_time = now.strftime("%H-%M-%S")
                writer.writerow([name, current_time])
    except:
        print(f"Error writing to {file_name}.")

    # Display frame and handle exit
    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()