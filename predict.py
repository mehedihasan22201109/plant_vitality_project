import cv2
import mediapipe as mp
import math
import pyautogui
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# =========================
# HAND TRACKING SETUP
# =========================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

screen_width, screen_height = pyautogui.size()

# =========================
# VOLUME CONTROL SETUP
# =========================

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None
)

volume = cast(interface, POINTER(IAudioEndpointVolume))

vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

# =========================
# LANDMARK IDS
# =========================

tip_ids = [4, 8, 12, 16, 20]

draw_points = []

# =========================
# MAIN LOOP
# =========================

while True:

    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img, 1)

    h, w, c = img.shape

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    finger_count = 0

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            lm_list = []

            for id, lm in enumerate(hand_landmarks.landmark):

                cx, cy = int(lm.x * w), int(lm.y * h)

                lm_list.append((cx, cy))

            # Draw landmarks
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # =========================
            # FINGER COUNT
            # =========================

            # Thumb
            if lm_list[tip_ids[0]][0] > lm_list[tip_ids[0] - 1][0]:
                finger_count += 1

            # Other fingers
            for i in range(1, 5):

                if lm_list[tip_ids[i]][1] < lm_list[tip_ids[i] - 2][1]:
                    finger_count += 1

            cv2.putText(
                img,
                f'Fingers: {finger_count}',
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3
            )

            # =========================
            # MOUSE CONTROL
            # Index finger controls mouse
            # =========================

            index_x, index_y = lm_list[8]

            screen_x = screen_width / w * index_x
            screen_y = screen_height / h * index_y

            pyautogui.moveTo(screen_x, screen_y)

            # =========================
            # CLICK CONTROL
            # Thumb + Index close = click
            # =========================

            thumb_x, thumb_y = lm_list[4]

            distance = math.hypot(index_x - thumb_x,
                                  index_y - thumb_y)

            if distance < 40:
                pyautogui.click()

                cv2.putText(
                    img,
                    'CLICK',
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3
                )

            # =========================
            # VOLUME CONTROL
            # Thumb + Middle finger distance
            # =========================

            middle_x, middle_y = lm_list[12]

            vol_distance = math.hypot(
                middle_x - thumb_x,
                middle_y - thumb_y
            )

            vol = min_vol + (vol_distance / 200) * (max_vol - min_vol)

            vol = max(min(vol, max_vol), min_vol)

            volume.SetMasterVolumeLevel(vol, None)

            cv2.putText(
                img,
                'Volume Control',
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2
            )

            # =========================
            # VIRTUAL DRAWING
            # Index finger drawing
            # =========================

            draw_points.append((index_x, index_y))

            for i in range(1, len(draw_points)):

                cv2.line(
                    img,
                    draw_points[i - 1],
                    draw_points[i],
                    (255, 0, 255),
                    5
                )

            # =========================
            # GESTURE DETECTION
            # =========================

            gesture = ""

            if finger_count == 0:
                gesture = "FIST"

            elif finger_count == 1:
                gesture = "ONE"

            elif finger_count == 2:
                gesture = "PEACE"

            elif finger_count == 5:
                gesture = "OPEN HAND"

            cv2.putText(
                img,
                f'Gesture: {gesture}',
                (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                3
            )

    cv2.imshow("AI Hand Gesture System", img)

    key = cv2.waitKey(1)

    # ESC to exit
    if key == 27:
        break

    # Press C to clear drawing
    if key == ord('c'):
        draw_points.clear()

cap.release()
cv2.destroyAllWindows()