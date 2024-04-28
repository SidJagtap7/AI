import cv2
import mediapipe as mpv
import time

cap = cv2.VideoCapture(0)


mpvHands = mpv.solutions.hands
hands = mpvHands.Hands()
mpvDraw = mpv.solutions.drawing_utils
cTime = 0
pTime = 0


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                if id ==0:
                    cv2.circle(img, (cx, cy), 20, (0, 0, 255), cv2.FILLED)
            mpvDraw.draw_landmarks(img, handLms, mpvHands.HAND_CONNECTIONS)





    cTime = time.time()
    FPS =1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(FPS)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)
    cv2.imshow("hand", img)
    cv2.waitKey(1)


