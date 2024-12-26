import cv2
import matplotlib.pyplot as plt

video_path = r'C:\Users\aya.ahmed\PycharmProjects\PythonProject1\.venv\WIN_20241226_11_57_07_Pro.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("Error opening video file. Check the file path.")

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frames per second: {fps}")
delay = int(1000 / fps)
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
nose_classifier = cv2.CascadeClassifier(r"C:\Users\aya.ahmed\Downloads\SimpleSensor-master\SimpleSensor-master\simplesensor\collection_modules\demographic_camera\classifiers\haarcascades\haarcascade_mcs_nose.xml")  # Update with the correct path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        roi_gray = gray_image[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_classifier.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

        noses = nose_classifier.detectMultiScale(roi_gray)
        for (nx, ny, nw, nh) in noses:
            cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 255, 255), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()