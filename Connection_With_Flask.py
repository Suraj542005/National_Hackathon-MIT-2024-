from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import time

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


def generate_frames():
    # cam = cv2.VideoCapture(0)
    cam = cv2.VideoCapture("Untitled video - Made with Clipchamp.mp4")
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    pTime = 0

    while True:
        success, img = cam.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    if id == 0:
                        cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# def generate_frames():
#     cam = cv2.VideoCapture(0)
#     mpHands = mp.solutions.hands
#     hands = mpHands.Hands()
#     mpDraw = mp.solutions.drawing_utils
#     pTime = 0
#
#     while True:
#         success, img = cam.read()
#         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         results = hands.process(imgRGB)
#         if results.multi_hand_landmarks:
#             for handLms in results.multi_hand_landmarks:
#                 for id, lm in enumerate(handLms.landmark):
#                     h, w, c = img.shape
#                     cx, cy = int(lm.x*w), int(lm.y*h)
#                     if id == 0:
#                         cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
#
#                 mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
#
#         cTime = time.time()
#         fps = 1/(cTime-pTime)
#         pTime = cTime
#
#         cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
#         cv2.putText(img, f"Coordinates: ({cx}, {cy})", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
#
#         ret, buffer = cv2.imencode('.jpg', img)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


app.run(debug=True)
