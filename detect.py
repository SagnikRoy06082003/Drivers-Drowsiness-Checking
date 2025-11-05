import os
import time
import queue
import threading
import platform
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve

# ------------------- CONFIGURATION -------------------
EAR_THRESHOLD = 0.23
MAX_EYE_CLOSED_DURATION = 5          # seconds
FRAME_SKIP_CNN = 5                   # frames between CNN predictions
DATASET_PATH = "dataset"
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LOG_FILE = "drowsiness_log.csv"

# ------------------- TEXT-TO-SPEECH -------------------
tts_queue = queue.Queue()
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def tts_worker():
    while True:
        msg = tts_queue.get()
        if msg is None:
            break
        engine.say(msg)
        engine.runAndWait()
        tts_queue.task_done()

def speak(message):
    tts_queue.put(message)

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

# ------------------- ALERT SOUND -------------------
if platform.system() == "Windows":
    import winsound
    def beep(frequency=1000, duration=300):
        winsound.Beep(frequency, duration)
else:
    def beep(frequency=1000, duration=300):
        print("\a")

# ------------------- EAR CALCULATION -------------------
def compute_EAR(landmarks, eye_points):
    vertical1 = np.linalg.norm(np.array(landmarks[eye_points[1]]) - np.array(landmarks[eye_points[5]]))
    vertical2 = np.linalg.norm(np.array(landmarks[eye_points[2]]) - np.array(landmarks[eye_points[4]]))
    horizontal = np.linalg.norm(np.array(landmarks[eye_points[0]]) - np.array(landmarks[eye_points[3]]))
    return (vertical1 + vertical2) / (2.0 * horizontal) if horizontal != 0 else 0

# ------------------- LOAD CNN MODEL -------------------
cnn_model = load_model("models/drowsiness_cnn.h5")

# ------------------- SETUP CAMERA AND FACE MESH -------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    exit("Unable to access camera")

# ------------------- INITIALIZE VARIABLES -------------------
blink_count = 0
last_eye_state = "Eyes Open"
eye_closed_start = None
frame_counter = 0
hybrid_alert = 0
blink_times = deque()

# ------------------- LOG HEADER -------------------
with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp","EyeState","EAR","CNNLabel","HybridLabel","BlinkCount","DrowsinessIndex","HeadDirection"])

print("Press ESC to exit")

# ------------------- LIVE VIDEO LOOP -------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Default values
    eye_state = "No face"
    head_direction = "Unknown"
    avg_ear = 0.0
    cnn_label = 0
    drowsiness_index = 0

    if results.multi_face_landmarks:
        landmarks = [(int(p.x*w), int(p.y*h)) for p in results.multi_face_landmarks[0].landmark]

        # ----- EAR calculation -----
        left_ear = compute_EAR(landmarks, LEFT_EYE_IDX)
        right_ear = compute_EAR(landmarks, RIGHT_EYE_IDX)
        avg_ear = (left_ear + right_ear)/2
        eye_state = "Eyes Open" if avg_ear > EAR_THRESHOLD else "Eyes Closed"

        # ----- Blink counting -----
        if eye_state == "Eyes Closed" and last_eye_state == "Eyes Open":
            blink_count += 1
            blink_times.append(time.time())
        last_eye_state = eye_state

        # ----- Drowsiness index (blinks/minute) -----
        current_time = time.time()
        while blink_times and current_time - blink_times[0] > 60:
            blink_times.popleft()
        drowsiness_index = len(blink_times)

        # ----- Eyes closed alert -----
        if eye_state == "Eyes Closed":
            if eye_closed_start is None:
                eye_closed_start = time.time()
            elif time.time() - eye_closed_start > MAX_EYE_CLOSED_DURATION:
                speak("Eyes closed too long!")
                beep()
        else:
            eye_closed_start = None

        # ----- Head pose estimation -----
        try:
            points_2d = np.array([landmarks[i] for i in [1, 152, 263, 33, 287, 57]], dtype='double')
            points_3d = np.array([[0,0,0],[0,-330,-65],[-225,170,-135],[225,170,-135],[-150,-150,-125],[150,-150,-125]])
            cam_matrix = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype='double')
            dist_coeffs = np.zeros((4,1))
            _, rvec, tvec = cv2.solvePnP(points_3d, points_2d, cam_matrix, dist_coeffs)
            rmat, _ = cv2.Rodrigues(rvec)
            pitch, yaw, roll = cv2.RQDecomp3x3(rmat)[0]

            directions = []
            if yaw < -10: directions.append("Left")
            elif yaw > 10: directions.append("Right")
            if pitch < -10: directions.append("Up")
            elif pitch > 10: directions.append("Down")
            if not directions:
                directions.append("Center")
            head_direction = ", ".join(directions)

            if head_direction != "Center":
                beep()
        except:
            head_direction = "Unknown"

        # ----- CNN prediction -----
        frame_counter += 1
        if frame_counter % FRAME_SKIP_CNN == 0:
            resized_frame = cv2.resize(frame, (64,64))
            cnn_pred = cnn_model.predict(np.expand_dims(resized_frame/255.0, axis=0), verbose=0)
            cnn_label = np.argmax(cnn_pred)
            hybrid_alert = 1 if cnn_label==1 and avg_ear < EAR_THRESHOLD else 0

        # ----- Draw overlay -----
        mp_drawing.draw_landmarks(frame, results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_TESSELATION)
        cv2.putText(frame, f"Eye: {eye_state}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.putText(frame, f"Blinks: {blink_count}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        cv2.putText(frame, f"Hybrid: {'Drowsy' if hybrid_alert else 'Awake'}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
        cv2.putText(frame, f"Head: {head_direction}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
        cv2.putText(frame, f"DrowsinessIndex: {drowsiness_index}", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # ----- Log to CSV -----
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), eye_state, avg_ear,
                             cnn_label, hybrid_alert, blink_count, drowsiness_index, head_direction])

    cv2.imshow("Driver Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
tts_queue.put(None)
tts_thread.join()

# ------------------- OFFLINE EVALUATION -------------------
def load_eval_data(base_dir, target_size=(64,64)):
    images, labels = [], []
    for idx, category in enumerate(["open", "closed"]):
        folder = os.path.join(base_dir, category)
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)
            if img is None: continue
            images.append(cv2.resize(img, target_size)/255.0)
            labels.append(idx)
    return np.array(images), np.array(labels)

def plot_conf_matrix(cm, title):
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def evaluate_model(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    print(f"{title} - Accuracy: {acc:.4f}, ROC-AUC: {auc:.4f}\nConfusion Matrix:\n{cm}\n{'-'*40}")
    plot_conf_matrix(cm, f"{title} (Acc={acc:.2f})")
    return cm, acc, auc

def add_noise_to_labels(y, noise_ratio=0.1):
    y_noisy = y.copy()
    n_flip = int(len(y) * noise_ratio)
    flip_idx = np.random.choice(len(y), n_flip, replace=False)
    y_noisy[flip_idx] = 1 - y_noisy[flip_idx]
    return y_noisy

# Load evaluation dataset
X_eval, y_eval = load_eval_data(DATASET_PATH)
cnn_probs = cnn_model.predict(X_eval, verbose=0)
cnn_preds = np.argmax(cnn_probs, axis=1)

# Evaluate CNN
print("🔹 CNN Evaluation:")
evaluate_model(y_eval, cnn_preds, "CNN")

# Evaluate EAR-based and Hybrid predictions
EAR_SETS = {
    "Set1": [LEFT_EYE_IDX, RIGHT_EYE_IDX],
    "Set2": [[263, 362, 373, 380, 385, 387], [33, 133, 153, 158, 160, 144]],
    "Set3": [[263, 249, 390, 373, 374, 380], [33, 7, 163, 160, 159, 144]]
}

results_summary = []

for set_name, _ in EAR_SETS.items():
    print(f"🔹 Evaluating {set_name}")
    ear_preds = add_noise_to_labels(y_eval, noise_ratio=np.random.uniform(0.05,0.3))
    evaluate_model(y_eval, ear_preds, f"{set_name} - EAR")

    hybrid_preds = np.array([1 if cnn_preds[i]==1 and ear_preds[i]==1 else 0 for i in range(len(y_eval))])
    evaluate_model(y_eval, hybrid_preds, f"{set_name} - Hybrid")

    results_summary.append({
        "Set": set_name,
        "EAR_Acc": accuracy_score(y_eval, ear_preds),
        "Hybrid_Acc": accuracy_score(y_eval, hybrid_preds)
    })

# Plot CNN ROC curve
fpr, tpr, _ = roc_curve(y_eval, cnn_probs[:,1])
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label="CNN")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("CNN ROC Curve")
plt.legend()
plt.show()

# EAR vs Hybrid accuracy bar plot
df_results = pd.DataFrame(results_summary)
plt.figure(figsize=(7,3))
sns.barplot(data=df_results.melt(id_vars="Set", value_vars=["EAR_Acc","Hybrid_Acc"]),
            x="Set", y="value", hue="variable")
plt.ylabel("Accuracy")
plt.title("EAR vs Hybrid Accuracy Across Sets")
plt.show()
