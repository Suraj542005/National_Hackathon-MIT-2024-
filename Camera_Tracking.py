import cv2
import mediapipe as mp
import time

cam = cv2.VideoCapture(0)
# For Hand Detection From Video.....
# cam = cv2.VideoCapture("Untitled video - Made with Clipchamp.mp4")
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0
# while cam.isOpened():
#     ret, frame = cam.read()
#     if cv2.waitKey(10) == ord('q'):
#         break
#     cv2.imshow("Camera", frame)

while True:
    success, img = cam.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id == 0:
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Resize the Image
    img = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)))  # Adjust the scaling factor as needed


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
