import cv2
import mediapipe as mp
import tkinter as tk
from angle import calculate_angle

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Auto-detect screen resolution
root = tk.Tk()
screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()
root.destroy()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

def get_coords(landmarks, index, w, h):
    lm = landmarks[index]
    return [int(lm.x * w), int(lm.y * h)]

def put_angle(frame, label, angle, position):
    cv2.putText(frame, f"{label}: {angle}°", position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

landmark_spec = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=4)
connection_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose, \
    mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb)
        hand_results = hands.process(rgb)

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark

            mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_spec,
                connection_spec)

            # GET ALL LANDMARKS
            nose        = get_coords(landmarks, mp_pose.PoseLandmark.NOSE.value, w, h)
            l_shoulder  = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value, w, h)
            r_shoulder  = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, w, h)
            l_elbow     = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW.value, w, h)
            r_elbow     = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW.value, w, h)
            l_wrist     = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_WRIST.value, w, h)
            r_wrist     = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value, w, h)
            l_hip       = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value, w, h)
            r_hip       = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value, w, h)
            l_knee      = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_KNEE.value, w, h)
            r_knee      = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE.value, w, h)
            l_ankle     = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE.value, w, h)
            r_ankle     = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE.value, w, h)
            l_heel      = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_HEEL.value, w, h)
            r_heel      = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HEEL.value, w, h)
            l_foot      = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value, w, h)
            r_foot      = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value, w, h)

            # Spine midpoints
            mid_shoulder = [(l_shoulder[0]+r_shoulder[0])//2, (l_shoulder[1]+r_shoulder[1])//2]
            mid_hip      = [(l_hip[0]+r_hip[0])//2,           (l_hip[1]+r_hip[1])//2]

            # Draw spine line
            cv2.line(frame, tuple(mid_shoulder), tuple(mid_hip), (0, 165, 255), 2)
            cv2.circle(frame, tuple(mid_hip), 5, (0, 165, 255), -1)

            # CALCULATE ANGLES
            left_elbow_angle     = calculate_angle(l_shoulder, l_elbow, l_wrist)
            right_elbow_angle    = calculate_angle(r_shoulder, r_elbow, r_wrist)
            left_shoulder_angle  = calculate_angle(l_hip, l_shoulder, l_elbow)
            right_shoulder_angle = calculate_angle(r_hip, r_shoulder, r_elbow)
            left_hip_angle       = calculate_angle(l_shoulder, l_hip, l_knee)
            right_hip_angle      = calculate_angle(r_shoulder, r_hip, r_knee)
            left_knee_angle      = calculate_angle(l_hip, l_knee, l_ankle)
            right_knee_angle     = calculate_angle(r_hip, r_knee, r_ankle)
            left_ankle_angle     = calculate_angle(l_knee, l_ankle, l_foot)
            right_ankle_angle    = calculate_angle(r_knee, r_ankle, r_foot)
            spine_angle          = calculate_angle(nose, mid_shoulder, mid_hip)

            # DISPLAY ANGLES (labels swapped due to mirror flip)
            put_angle(frame, "RE",  left_elbow_angle,      tuple(l_elbow))
            put_angle(frame, "LE",  right_elbow_angle,     tuple(r_elbow))
            put_angle(frame, "RS",  left_shoulder_angle,   tuple(l_shoulder))
            put_angle(frame, "LS",  right_shoulder_angle,  tuple(r_shoulder))
            put_angle(frame, "RH",  left_hip_angle,        tuple(l_hip))
            put_angle(frame, "LH",  right_hip_angle,       tuple(r_hip))
            put_angle(frame, "RK",  left_knee_angle,       tuple(l_knee))
            put_angle(frame, "LK",  right_knee_angle,      tuple(r_knee))
            put_angle(frame, "RA",  left_ankle_angle,      tuple(l_ankle))
            put_angle(frame, "LA",  right_ankle_angle,     tuple(r_ankle))
            put_angle(frame, "SP",  spine_angle,           tuple(mid_shoulder))

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

        frame = cv2.resize(frame, (screen_w, screen_h))
        cv2.namedWindow("Form Coach", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Form Coach", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Form Coach", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if cv2.getWindowProperty("Form Coach", cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()