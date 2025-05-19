import cv2
import time
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("/Users/bineesh/Downloads/face_recognition_and_door_lock-main/haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("/Users/bineesh/Downloads/face_recognition_and_door_lock-main/trainer.yml")
name_list = ["","Unknown","Unknown","Unknown","Unknown","Unknown","Unknown"]
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        face_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 6)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 5)
        if confidence > 50:
            cv2.putText(frame, name_list[face_id], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Unknown", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == ord('o') and confidence > 50:
        time.sleep(10)
    if key == ord("q"):
        break
video.release()
cv2.destroyAllWindows()
