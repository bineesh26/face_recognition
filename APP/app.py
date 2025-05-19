from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import numpy as np
import time

app = Flask(__name__)

faceDetect = cv2.CascadeClassifier('/Users/bineesh/Downloads/face_recognition/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("/Users/bineesh/Downloads/face_recognition/trainer.yml")
name_list = ["", "Unknown", "Unknown", "Unknown", "Unknown", "Unknown"]

video = cv2.VideoCapture(0)
sampleNum = 0
id = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_capture', methods=['POST'])
def start_capture():
    global id, sampleNum
    id = request.form['user_id']
    sampleNum = 0
    return redirect(url_for('capture'))

@app.route('/capture')
def capture():
    return render_template('capture.html')

def generate_frames():
    global sampleNum
    while True:
        ret, img = video.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            sampleNum += 1
            cv2.imwrite(f"/Users/bineesh/Downloads/face_recognition/Datas/data.{id}.{sampleNum}.jpg", gray[y:y + h, x:x + w])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)
            cv2.waitKey(100)
        if sampleNum > 500:
            break
        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    video.release()
    cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognize')
def recognize():
    return render_template('recognize.html')

def generate_recognition_frames():
    while True:
        ret, frame = video.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 6)
            if confidence > 50:
                cv2.putText(frame, name_list[face_id], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "BINEESH", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    video.release()
    cv2.destroyAllWindows()

@app.route('/recognition_feed')
def recognition_feed():
    return Response(generate_recognition_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=2026)
