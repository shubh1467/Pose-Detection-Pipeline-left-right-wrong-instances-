import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import math

# =====================================================
# MODELS
# =====================================================

pose_model = YOLO("yolo11x-pose.pt")

bat_model = YOLO(
    "/home/ds-khel/pose_detection/best.pt"
)

BAT_CLASS_ID = 0

# =====================================================
# THRESHOLDS (CALIBRATED FOR CRICKET STANCE)
# =====================================================

# Body ratios for front/side classification
SH_HIP_FRONT = 1.42  #1.48
SH_HIP_SIDE  = 1.38 #1.42

SH_TORSO_FRONT = 0.83
SH_TORSO_SIDE  = 0.75

SHOULDER_WIDTH_THRESHOLD = 65.937

# =====================================================
# KEYPOINTS
# =====================================================

NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
L_SHOULDER = 5
R_SHOULDER = 6
L_ELBOW = 7
R_ELBOW = 8
L_WRIST = 9
R_WRIST = 10
L_HIP = 11
R_HIP = 12
L_KNEE = 13
R_KNEE = 14
L_ANKLE = 15
R_ANKLE = 16

# =====================================================
# VIDEO PATHS - MODIFY THESE
# =====================================================

video_input = "/home/ds-khel/pose_detection/hit1.mp4"      
video_output = "/home/ds-khel/pose_detection/output_measurement_hit1_x5.mp4" 

# =====================================================
# SKELETON CONNECTIONS
# =====================================================

SKELETON = [
    (5,6),   # shoulders
    (5,7), (7,9),   # left arm
    (6,8), (8,10),  # right arm
    (5,11), (6,12), # torso
    (11,12),        # hips
    (11,13), (13,15),  # left leg
    (12,14), (14,16)   # right leg
]

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def dist(a, b):
    """Euclidean distance between two points"""
    return np.linalg.norm(np.array(a) - np.array(b))

def angle_between(p1, p2, p3):
    """Calculate angle between three points (p1-p2-p3)"""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    cos_angle = dot / (norm1 * norm2)
    cos_angle = max(-1, min(1, cos_angle))
    
    return math.degrees(math.acos(cos_angle))

def visible(kps, i):
    """Check if keypoint is visible with confidence > 0.75"""
    return kps[i][2] > 0.75

def get_shoulder_line_angle(kps):
    """Calculate angle of shoulder line relative to horizontal"""
    if not (visible(kps, L_SHOULDER) and visible(kps, R_SHOULDER)):
        return None
    
    dx = kps[R_SHOULDER][0] - kps[L_SHOULDER][0]
    dy = kps[R_SHOULDER][1] - kps[L_SHOULDER][1]
    
    angle = math.degrees(math.atan2(dy, dx))
    return abs(angle)

def get_spine_angle(kps):
    """Calculate angle of spine (mid-shoulder to mid-hip) relative to vertical"""
    if not (visible(kps, L_SHOULDER) and visible(kps, R_SHOULDER) and
            visible(kps, L_HIP) and visible(kps, R_HIP)):
        return None
    
    shoulder_mid = (
        (kps[L_SHOULDER][0] + kps[R_SHOULDER][0]) / 2,
        (kps[L_SHOULDER][1] + kps[R_SHOULDER][1]) / 2
    )
    
    hip_mid = (
        (kps[L_HIP][0] + kps[R_HIP][0]) / 2,
        (kps[L_HIP][1] + kps[R_HIP][1]) / 2
    )
    
    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    
    angle = math.degrees(math.atan2(dx, dy))
    return abs(angle)

def get_body_asymmetry(kps):
    """Calculate left-right asymmetry score"""
    if not (visible(kps, L_SHOULDER) and visible(kps, R_SHOULDER) and
            visible(kps, L_HIP) and visible(kps, R_HIP)):
        return None
    
    shoulder_height_diff = abs(kps[L_SHOULDER][1] - kps[R_SHOULDER][1])
    hip_height_diff = abs(kps[L_HIP][1] - kps[R_HIP][1])
    
    # Normalize by shoulder width
    shoulder_width = dist(kps[L_SHOULDER][:2], kps[R_SHOULDER][:2])
    if shoulder_width > 0:
        shoulder_asymmetry = shoulder_height_diff / shoulder_width
        hip_asymmetry = hip_height_diff / shoulder_width
    else:
        return None
    
    return (shoulder_asymmetry + hip_asymmetry) / 2

def get_limb_angles(kps):
    """Calculate key joint angles for stance analysis"""
    angles = {}
    
    # Left elbow angle
    if all(visible(kps, i) for i in [L_SHOULDER, L_ELBOW, L_WRIST]):
        angles['left_elbow'] = angle_between(
            kps[L_SHOULDER][:2], kps[L_ELBOW][:2], kps[L_WRIST][:2]
        )
    
    # Right elbow angle
    if all(visible(kps, i) for i in [R_SHOULDER, R_ELBOW, R_WRIST]):
        angles['right_elbow'] = angle_between(
            kps[R_SHOULDER][:2], kps[R_ELBOW][:2], kps[R_WRIST][:2]
        )
    
    # Left knee angle
    if all(visible(kps, i) for i in [L_HIP, L_KNEE, L_ANKLE]):
        angles['left_knee'] = angle_between(
            kps[L_HIP][:2], kps[L_KNEE][:2], kps[L_ANKLE][:2]
        )
    
    # Right knee angle
    if all(visible(kps, i) for i in [R_HIP, R_KNEE, R_ANKLE]):
        angles['right_knee'] = angle_between(
            kps[R_HIP][:2], kps[R_KNEE][:2], kps[R_ANKLE][:2]
        )
    
    return angles

# =====================================================
# DRAW FUNCTIONS
# =====================================================

def draw_pose(frame, kps):
    """Draw pose skeleton on frame"""
    for kp in kps:
        x, y, c = kp
        if c > 0.5:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    
    for a, b in SKELETON:
        if kps[a][2] > 0.5 and kps[b][2] > 0.5:
            p1 = (int(kps[a][0]), int(kps[a][1]))
            p2 = (int(kps[b][0]), int(kps[b][1]))
            cv2.line(frame, p1, p2, (0, 255, 0), 3)

def draw_bat_mask(frame, result):
    """Draw orange mask on detected bat"""
    if result.boxes is None:
        return frame, None
    
    vis = frame.copy()
    overlay = frame.copy()
    bat_center = None

    for box in result.boxes:
        if int(box.cls[0]) != BAT_CLASS_ID:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Draw filled rectangle as mask
        # cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), -1)  # COMMENTED: bat orange mask

        # Compute center
        bat_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    # Blend overlay
    # vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)  # COMMENTED: bat mask blend
    return vis, bat_center


# =====================================================
# DEBUG OVERLAY FUNCTION
# =====================================================

def overlay_debug(vis, feats, label, x=20, y=80):
    """Draw all decision features on screen"""

    if feats is None:
        cv2.putText(vis, "NO FEATURES", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        return

    line_height = 42
    i = 0

    def put(text, color=(255,255,255), scale=0.6, thickness=2):
        nonlocal i
        cv2.putText(
            vis,
            text,
            (x, y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness
        )
        i += 1

    # =============================
    # LABEL ONLY
    # =============================
    color_map = {
        "FRONT VIEW":     (0, 255, 0),
        "SIDE VIEW":      (0, 165, 255),
        "WRONG INSTANCE": (0, 0, 255)
    }
    label_color = color_map.get(label, (255, 255, 255))

    cv2.putText(
        vis,
        f"Label: {label}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        label_color,
        3
    )
    i += 1  # keep line counter in sync

    # =============================
    # BASIC METRICS  — COMMENTED OUT
    # =============================
    # put(f"Shoulder: {feats['shoulder']:.1f}")
    # put(f"Hip: {feats['hip']:.1f}")
    # put(f"Torso: {feats['torso']:.1f}")

    # put(f"SH/HIP: {feats['sh_hip']:.2f}")
    # put(f"SH/TORSO: {feats['sh_torso']:.2f}")

    # =============================
    # FACE FEATURES  — COMMENTED OUT
    # =============================
    # put(f"Eyes: {feats['eyes']}")
    # put(f"Ears: {feats['ears']}")
    # put(f"Nose visible: {feats['nose']}")
    # put(f"Nose symmetry: {feats['nose_symmetry']:.2f}")

    # =============================
    # ANGLES  — COMMENTED OUT
    # =============================
    # if feats["shoulder_angle"] is not None:
    #     put(f"Shoulder angle: {feats['shoulder_angle']:.1f}")

    # if feats["spine_angle"] is not None:
    #     put(f"Spine angle: {feats['spine_angle']:.1f}")

    # if feats["asymmetry_score"] is not None:
    #     put(f"Asymmetry: {feats['asymmetry_score']:.3f}")

    # =============================
    # LIMB ANGLES  — COMMENTED OUT
    # =============================
    # if feats["left_elbow_angle"] is not None:
    #     put(f"L-Elbow: {feats['left_elbow_angle']:.1f}")

    # if feats["right_elbow_angle"] is not None:
    #     put(f"R-Elbow: {feats['right_elbow_angle']:.1f}")

    # if feats["left_knee_angle"] is not None:
    #     put(f"L-Knee: {feats['left_knee_angle']:.1f}")

    # if feats["right_knee_angle"] is not None:
    #     put(f"R-Knee: {feats['right_knee_angle']:.1f}")

    # =============================
    # QUALITY  — COMMENTED OUT
    # =============================
    # put(f"Pose quality: {feats['pose_quality']:.2f}")

    # =============================
    # DECISION FLAGS  — COMMENTED OUT
    # =============================
    # put("---- DECISION FLAGS ----", (0,255,255))

    # body_front = feats["sh_hip"] > SH_HIP_FRONT
    # body_side  = feats["sh_hip"] < SH_HIP_SIDE

    # face_front = (feats["eyes"] == 2 and feats["ears"] == 2 and feats["nose_symmetry"] > 0.80)
    # face_side  = (feats["eyes"] == 1 and feats["ears"] == 1 and feats["nose_symmetry"] < 0.50)

    # put(f"Body Front: {body_front}", (0,255,0) if body_front else (0,0,255))
    # put(f"Body Side: {body_side}", (0,255,0) if body_side else (0,0,255))

    # put(f"Face Front: {face_front}", (0,255,0) if face_front else (0,0,255))
    # put(f"Face Side: {face_side}", (0,255,0) if face_side else (0,0,255))

def get_person_score(kps, box, bat_center):
    """Score person based on proximity to bat"""
    if bat_center is None:
        return 999999
    
    score = 0
    wrists = []
    
    if kps[L_WRIST][2] > 0.5:
        wrists.append(kps[L_WRIST][:2])
    if kps[R_WRIST][2] > 0.5:
        wrists.append(kps[R_WRIST][:2])
    
    if len(wrists) > 0:
        d = min(np.linalg.norm(np.array(w) - bat_center) for w in wrists)
        score += d * 0.6
    
    return score

# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_features(kps, bat_center=None, frame_width=0):
    """Extract comprehensive features for classification"""
    
    # Check visibility of key body parts
    required_keypoints = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, NOSE]
    for kp in required_keypoints:
        if kps[kp][2] < 0.5:
            return None
    
    # Basic measurements
    shoulder = dist(kps[L_SHOULDER][:2], kps[R_SHOULDER][:2])
    hip = dist(kps[L_HIP][:2], kps[R_HIP][:2])
    
    shoulder_mid = (
        (kps[L_SHOULDER][0] + kps[R_SHOULDER][0]) / 2,
        (kps[L_SHOULDER][1] + kps[R_SHOULDER][1]) / 2
    )
    
    hip_mid = (
        (kps[L_HIP][0] + kps[R_HIP][0]) / 2,
        (kps[L_HIP][1] + kps[R_HIP][1]) / 2
    )
    
    torso = dist(shoulder_mid, hip_mid)
    
    if torso < 5:
        return None
    
    # Face visibility
    eyes = 0
    ears = 0
    
    if visible(kps, LEFT_EYE):
        eyes += 1
    if visible(kps, RIGHT_EYE):
        eyes += 1
    if visible(kps, LEFT_EAR):
        ears += 1
    if visible(kps, RIGHT_EAR):
        ears += 1
    
    nose_visible = visible(kps, NOSE)
    
    # Nose symmetry
    nose_symmetry = 0
    if (visible(kps, NOSE) and visible(kps, LEFT_EYE) and visible(kps, RIGHT_EYE)):
        dl = dist(kps[NOSE][:2], kps[LEFT_EYE][:2])
        dr = dist(kps[NOSE][:2], kps[RIGHT_EYE][:2])
        if max(dl, dr) > 0:
            nose_symmetry = min(dl, dr) / max(dl, dr)
    
    # Asymmetry and angle features
    shoulder_angle = get_shoulder_line_angle(kps)
    spine_angle = get_spine_angle(kps)
    asymmetry_score = get_body_asymmetry(kps)
    limb_angles = get_limb_angles(kps)
    
    # Left/Right indicators
    left_shoulder_higher = False
    if visible(kps, L_SHOULDER) and visible(kps, R_SHOULDER):
        left_shoulder_higher = kps[L_SHOULDER][1] < kps[R_SHOULDER][1]
    
    left_hip_higher = False
    if visible(kps, L_HIP) and visible(kps, R_HIP):
        left_hip_higher = kps[L_HIP][1] < kps[R_HIP][1]
    
    # Pose quality
    pose_quality = np.mean([
        kps[L_SHOULDER][2], kps[R_SHOULDER][2],
        kps[L_HIP][2], kps[R_HIP][2],
        kps[NOSE][2]
    ])
    
    return {
        "shoulder": shoulder,
        "hip": hip,
        "torso": torso,
        "sh_hip": shoulder / hip if hip > 0 else 0,
        "sh_torso": shoulder / torso if torso > 0 else 0,
        "eyes": eyes,
        "ears": ears,
        "nose": nose_visible,
        "nose_symmetry": nose_symmetry,
        "pose_quality": pose_quality,
        "shoulder_angle": shoulder_angle,
        "spine_angle": spine_angle,
        "asymmetry_score": asymmetry_score,
        "left_shoulder_higher": left_shoulder_higher,
        "left_hip_higher": left_hip_higher,
        "left_elbow_angle": limb_angles.get('left_elbow', None),
        "right_elbow_angle": limb_angles.get('right_elbow', None),
        "left_knee_angle": limb_angles.get('left_knee', None),
        "right_knee_angle": limb_angles.get('right_knee', None)
    }

# =====================================================
# SIMPLIFIED CLASSIFICATION (FRONT/SIDE ONLY)
# =====================================================

def classify_view(feats):
    """Simplified classification: FRONT VIEW, SIDE VIEW, or WRONG INSTANCE"""
    
    if feats is None:
        return "WRONG INSTANCE"
    
    # ==========================================
    # REJECTION CRITERIA (WRONG INSTANCE)
    # ==========================================
    
    # Missing face features
    if feats["eyes"] == 0 or feats["ears"] == 0:
        return "WRONG INSTANCE"
    
    # Poor pose quality
    if feats["pose_quality"] < 0.75:
        return "WRONG INSTANCE"
    
    # Ambiguous shoulder width
    if 110 <= feats["shoulder"] <= 140:
        return "WRONG INSTANCE"
    
    # Ambiguous nose symmetry
    if 0.50 <= feats["nose_symmetry"] <= 0.80:
        return "WRONG INSTANCE"
    
    # Face visibility score
    face_score = feats["eyes"] + feats["ears"] + int(feats["nose"])
    if face_score < 3:
        return "WRONG INSTANCE"

    # Body asymmetry (hand raise, imbalance)
    if feats["asymmetry_score"] is not None:
        if feats["asymmetry_score"] > 0.25:
            return "WRONG INSTANCE"

    # Elbow extremes (raised bat / shot motion)
    if feats["right_elbow_angle"] is not None:
        if feats["right_elbow_angle"] < 40 or feats["right_elbow_angle"] > 160:
            return "WRONG INSTANCE"

    if feats["left_elbow_angle"] is not None:
        if feats["left_elbow_angle"] < 40 or feats["left_elbow_angle"] > 160:
            return "WRONG INSTANCE"

    # Spine tilt
    if feats["spine_angle"] is not None:
        if feats["spine_angle"] > 20:
            return "WRONG INSTANCE"
    
    # Head-body orientation conflict
    body_front = (feats["sh_hip"] > SH_HIP_FRONT)
    body_side = (feats["sh_hip"] < SH_HIP_SIDE)
    
    face_front = (feats["eyes"] == 2 and feats["ears"] == 2 and feats["nose_symmetry"] > 0.80)
    face_side = (feats["eyes"] == 1 and feats["ears"] == 1 and feats["nose_symmetry"] < 0.50)
    
    if (body_front and face_side) or (body_side and face_front):
        return "WRONG INSTANCE"
    
    # ==========================================
    # FRONT/SIDE SCORING
    # ==========================================
    
    front = 0
    side = 0
    
    # Body ratio checks
    if feats["sh_hip"] > SH_HIP_FRONT:
        front += 1
    elif feats["sh_hip"] < SH_HIP_SIDE:
        side += 1
    
    if feats["sh_torso"] > SH_TORSO_FRONT:
        front += 1
    elif feats["sh_torso"] < SH_TORSO_SIDE:
        side += 1
    
    # Shoulder vs hip comparison
    if feats["shoulder"] > feats["hip"] * 1.12:
        front += 1
    else:
        side += 1
    
    # Shoulder width
    if feats["shoulder"] > SHOULDER_WIDTH_THRESHOLD:
        side += 1
    else:
        front += 1
    
    # Face features
    if feats["eyes"] == 2:
        front += 2
    elif feats["eyes"] == 1:
        side += 2
    
    if feats["ears"] == 2:
        front += 2
    elif feats["ears"] == 1:
        side += 1
    
    if feats["nose"]:
        front += 1
    else:
        side += 1
    
    if feats["nose_symmetry"] > 0.80:
        front += 2
    elif feats["nose_symmetry"] < 0.50:
        side += 2
    
    # Elbow angle analysis
    if feats["right_elbow_angle"] is not None:
        if 60 < feats["right_elbow_angle"] < 120:
            if front >= side:
                front += 1
    
    if feats["left_elbow_angle"] is not None:
        if 60 < feats["left_elbow_angle"] < 120:
            if front >= side:
                front += 1
    
    # Knee angle analysis
    if feats["right_knee_angle"] is not None:
        if 150 < feats["right_knee_angle"] < 170:
            if front >= side:
                front += 1
    
    if feats["left_knee_angle"] is not None:
        if 150 < feats["left_knee_angle"] < 170:
            if front >= side:
                front += 1
    
    # ==========================================
    # FINAL CLASSIFICATION
    # ==========================================
    
    if front >= 5:
        return "FRONT VIEW"
    elif side >= 5:
        return "SIDE VIEW"
    else:
        return "WRONG INSTANCE"

# =====================================================
# MAIN PROCESSING LOOP
# =====================================================

cap = cv2.VideoCapture(video_input)

w = int(cap.get(3))
h = int(cap.get(4))

fps = int(cap.get(5))

if fps <= 0:
    fps = 30

out = cv2.VideoWriter(
    video_output,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (w, h)
)

locked_track_id = None
votes = deque(maxlen=int(fps * 1.0))  # 1.0 sec window

# Statistics collection
view_counts = {
    "FRONT VIEW": 0,
    "SIDE VIEW": 0,
    "WRONG INSTANCE": 0
}

# =====================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    vis = frame.copy()
    
    # Bat detection
    bat_res = bat_model(frame, verbose=False)[0]
    vis, bat_center = draw_bat_mask(vis, bat_res)

    # COMMENTED: bat center dot
    # if bat_center is not None:
    #     cv2.circle(vis, (int(bat_center[0]), int(bat_center[1])), 10, (255, 0, 0), -1)
    
    # Pose tracking
    res = pose_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )[0]
    
    selected_idx = None
    label = "WRONG INSTANCE"
    
    if (res.boxes is not None and 
        res.boxes.id is not None and 
        res.keypoints is not None):
        
        ids = res.boxes.id.cpu().numpy().astype(int)
        boxes = res.boxes.xyxy.cpu().numpy()
        people = res.keypoints.data.cpu().numpy()
        
        # Person selection logic
        if locked_track_id is None:
            best_score = 999999
            for i in range(len(ids)):
                score = get_person_score(people[i], boxes[i], bat_center)
                if score < best_score:
                    best_score = score
                    selected_idx = i
                    locked_track_id = ids[i]
        else:
            for i in range(len(ids)):
                if ids[i] == locked_track_id:
                    selected_idx = i
                    break
        
        if selected_idx is not None:
            # Draw pose skeleton
            draw_pose(vis, people[selected_idx])
            
            # Extract features
            feats = extract_features(people[selected_idx], bat_center, w)
            
            # Classify view
            label = classify_view(feats)

            # Draw debug info (only "Label: {text}" is visible — rest is commented inside)
            overlay_debug(vis, feats, label)
            
            # Update statistics
            if label in view_counts:
                view_counts[label] += 1
            
            # Draw bounding box with color matching label
            color_map = {
                "FRONT VIEW":     (0, 255,   0),   # green
                "SIDE VIEW":      (0, 165, 255),   # orange
                "WRONG INSTANCE": (0,   0, 255)    # red
            }
            bbox_color = color_map.get(label, (255, 255, 255))
            x1, y1, x2, y2 = map(int, boxes[selected_idx])
            cv2.rectangle(vis, (x1, y1), (x2, y2), bbox_color, 3)
    
    # Stabilize predictions with voting
    votes.append(label)
    stable = max(set(votes), key=votes.count)

    # COMMENTED: large stable label top-left (the first coloured front/side/wrong text)
    # color_map = {
    #     "FRONT VIEW":     (0, 255,   0),
    #     "SIDE VIEW":      (0, 165, 255),
    #     "WRONG INSTANCE": (0,   0, 255)
    # }
    # color = color_map.get(stable, (255, 255, 255))
    # cv2.putText(
    #     vis,
    #     stable,
    #     (30, 50),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     1.2,
    #     color,
    #     3
    # )

    # COMMENTED: bat center dot (duplicate guard)
    # if bat_center is not None:
    #     cv2.circle(vis, (int(bat_center[0]), int(bat_center[1])), 10, (255, 0, 0), -1)
    
    out.write(vis)

# =====================================================
# CLEANUP AND STATISTICS
# =====================================================

cap.release()
out.release()

# Final classification (most frequent)
final_label = max(set(votes), key=votes.count)

print("\n" + "="*50)
print("POSE CLASSIFICATION RESULTS")
print("="*50)
print(f"Final classification: {final_label}")
print(f"\nFrame-by-frame statistics:")
for view, count in view_counts.items():
    if count > 0:
        percentage = (count / len(votes)) * 100
        print(f"  {view}: {count} frames ({percentage:.1f}%)")
print(f"\nOutput saved: {video_output}")