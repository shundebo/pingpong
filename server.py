import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify
import random
import webbrowser

app = Flask(__name__)

# Camera setup
cap = cv2.VideoCapture(0)

def detect_hand():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale and apply threshold
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            hand_position = y + h // 2
        else:
            hand_position = None
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def log_message(message):
    with open("log.txt", "a") as log_file:
        log_file.write(message + "\n")
    print(message)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_hand(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/hand_position', methods=['GET'])
def hand_position():
    ret, frame = cap.read()
    if not ret:
        return jsonify({"hand_position": None})

    # Convert to grayscale and apply threshold
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        hand_position = y + h // 2
    else:
        hand_position = None
    
    return jsonify({"hand_position": hand_position})

if __name__ == '__main__':
    log_message("Starting server...")
    webbrowser.open_new("http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)