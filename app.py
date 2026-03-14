from coach import analyze_posture
from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import threading
from angle import calculate_angle

app = Flask(__name__)
CORS(app)

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

angle_data = {}
lock = threading.Lock()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

def get_coords(landmarks, index, w, h):
    lm = landmarks[index]
    return [int(lm.x * w), int(lm.y * h)]

def put_angle(frame, label, angle, position):
    cv2.putText(frame, f"{label}: {angle}°", position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

def generate_frames():
    global angle_data
    frame_count = 0
    last_pose_results = None
    last_hand_results = None
    cached_angles = {}
    cached_coords = {}

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose, \
        mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Only run MediaPipe every other frame
            if frame_count % 2 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                last_pose_results = pose.process(rgb)
                last_hand_results = hands.process(rgb)

                if last_pose_results.pose_landmarks:
                    landmarks = last_pose_results.pose_landmarks.landmark

                    nose       = get_coords(landmarks, mp_pose.PoseLandmark.NOSE.value, w, h)
                    l_shoulder = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value, w, h)
                    r_shoulder = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, w, h)
                    l_elbow    = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW.value, w, h)
                    r_elbow    = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW.value, w, h)
                    l_wrist    = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_WRIST.value, w, h)
                    r_wrist    = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value, w, h)
                    l_hip      = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value, w, h)
                    r_hip      = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value, w, h)
                    l_knee     = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_KNEE.value, w, h)
                    r_knee     = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE.value, w, h)
                    l_ankle    = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE.value, w, h)
                    r_ankle    = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE.value, w, h)
                    l_foot     = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value, w, h)
                    r_foot     = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value, w, h)

                    mid_shoulder = [(l_shoulder[0]+r_shoulder[0])//2, (l_shoulder[1]+r_shoulder[1])//2]
                    mid_hip      = [(l_hip[0]+r_hip[0])//2, (l_hip[1]+r_hip[1])//2]

                    cached_coords = {
                        "nose": nose, "l_shoulder": l_shoulder, "r_shoulder": r_shoulder,
                        "l_elbow": l_elbow, "r_elbow": r_elbow,
                        "l_wrist": l_wrist, "r_wrist": r_wrist,
                        "l_hip": l_hip, "r_hip": r_hip,
                        "l_knee": l_knee, "r_knee": r_knee,
                        "l_ankle": l_ankle, "r_ankle": r_ankle,
                        "l_foot": l_foot, "r_foot": r_foot,
                        "mid_shoulder": mid_shoulder, "mid_hip": mid_hip
                    }

                    cached_angles = {
                        "left_elbow":     calculate_angle(l_shoulder, l_elbow, l_wrist),
                        "right_elbow":    calculate_angle(r_shoulder, r_elbow, r_wrist),
                        "left_shoulder":  calculate_angle(l_hip, l_shoulder, l_elbow),
                        "right_shoulder": calculate_angle(r_hip, r_shoulder, r_elbow),
                        "left_hip":       calculate_angle(l_shoulder, l_hip, l_knee),
                        "right_hip":      calculate_angle(r_shoulder, r_hip, r_knee),
                        "left_knee":      calculate_angle(l_hip, l_knee, l_ankle),
                        "right_knee":     calculate_angle(r_hip, r_knee, r_ankle),
                        "left_ankle":     calculate_angle(l_knee, l_ankle, l_foot),
                        "right_ankle":    calculate_angle(r_knee, r_ankle, r_foot),
                        "spine":          calculate_angle(nose, mid_shoulder, mid_hip)
                    }

                    with lock:
                        angle_data = cached_angles

            # Always draw from cached results on every frame
            if last_pose_results and last_pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    last_pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

            if cached_coords and cached_angles:
                cv2.line(frame, tuple(cached_coords["mid_shoulder"]), tuple(cached_coords["mid_hip"]), (0, 165, 255), 2)
                put_angle(frame, "RE", cached_angles["left_elbow"],     tuple(cached_coords["l_elbow"]))
                put_angle(frame, "LE", cached_angles["right_elbow"],    tuple(cached_coords["r_elbow"]))
                put_angle(frame, "RS", cached_angles["left_shoulder"],  tuple(cached_coords["l_shoulder"]))
                put_angle(frame, "LS", cached_angles["right_shoulder"], tuple(cached_coords["r_shoulder"]))
                put_angle(frame, "RH", cached_angles["left_hip"],       tuple(cached_coords["l_hip"]))
                put_angle(frame, "LH", cached_angles["right_hip"],      tuple(cached_coords["r_hip"]))
                put_angle(frame, "RK", cached_angles["left_knee"],      tuple(cached_coords["l_knee"]))
                put_angle(frame, "LK", cached_angles["right_knee"],     tuple(cached_coords["r_knee"]))
                put_angle(frame, "RA", cached_angles["left_ankle"],     tuple(cached_coords["l_ankle"]))
                put_angle(frame, "LA", cached_angles["right_ankle"],    tuple(cached_coords["r_ankle"]))
                put_angle(frame, "SP", cached_angles["spine"],          tuple(cached_coords["mid_shoulder"]))

            if last_hand_results and last_hand_results.multi_hand_landmarks:
                for hand_landmarks in last_hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS)

            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 50]
            _, buffer = cv2.imencode('.jpg', frame, encode_params)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/angles')
def angles():
    with lock:
        return jsonify(angle_data)

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

@app.route('/feedback')
def feedback():
    with lock:
        angles = angle_data.copy()
    result = analyze_posture(angles)
    return jsonify({"feedback": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)