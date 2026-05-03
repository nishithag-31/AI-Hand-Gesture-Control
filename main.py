import cv2
import mediapipe as mp
import random

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
draw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_points = []

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            points = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                points.append((cx, cy))

            hand_points.append(points)
            draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # 🔥 DRAW LINES BETWEEN TWO HANDS
    if len(hand_points) == 2:
        hand1 = hand_points[0]
        hand2 = hand_points[1]

        # Connect fingertip points (4,8,12,16,20)
        tips = [4, 8, 12, 16, 20]

        for i in tips:
            x1, y1 = hand1[i]
            x2, y2 = hand2[i]

            color = (random.randint(0,255),
                     random.randint(0,255),
                     random.randint(0,255))

            cv2.line(img, (x1, y1), (x2, y2), color, 3)

    cv2.imshow("AI Gesture Art", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()