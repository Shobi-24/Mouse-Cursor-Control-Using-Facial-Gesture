import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
import math
import matplotlib.pyplot as plt
pyautogui.FAILSAFE = False
CALIB_TIME = 4.0                
BLINK_WINDOW = 0.85             
SMOOTHING_ALPHA = 0.16
SENSITIVITY_CHOICE = 2
SENS_MAP = {1: 0.8, 2: 1.6, 3: 2.8}
SENSITIVITY = SENS_MAP.get(SENSITIVITY_CHOICE, 1.6)
SCROLL_STEP_AMOUNT = 200
SCROLL_NEUTRAL_MARGIN = 0.02
SCROLL_TRIGGER_THRESHOLD = 0.035
READY_DELAY = 2.0               
EYEBROW_HOLD = 1.2              
EYEBROW_FACTOR = 1.25           
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_OUTER = [61, 291]         
MOUTH_INNER_TOP = 13
MOUTH_INNER_BOTTOM = 14
NOSE_TIP = 1
LEFT_EYEBROW_POINT = 105
RIGHT_EYEBROW_POINT = 334
LEFT_EYE_REF = 33
RIGHT_EYE_REF = 263
times = []
ears = []
mars = []
nose_ys = []
events = []
def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])
def eye_aspect_ratio(landmarks, eye_idxs, frame_w, frame_h):
    pts = [(int(landmarks[i][0]*frame_w), int(landmarks[i][1]*frame_h)) for i in eye_idxs]
    A = dist(pts[1], pts[5])
    B = dist(pts[2], pts[4])
    C = dist(pts[0], pts[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)
def mouth_aspect_ratio(landmarks, frame_w, frame_h):
    top = (int(landmarks[MOUTH_INNER_TOP][0]*frame_w), int(landmarks[MOUTH_INNER_TOP][1]*frame_h))
    bottom = (int(landmarks[MOUTH_INNER_BOTTOM][0]*frame_w), int(landmarks[MOUTH_INNER_BOTTOM][1]*frame_h))
    left = (int(landmarks[MOUTH_OUTER[0]][0]*frame_w), int(landmarks[MOUTH_OUTER[0]][1]*frame_h))
    right = (int(landmarks[MOUTH_OUTER[1]][0]*frame_w), int(landmarks[MOUTH_OUTER[1]][1]*frame_h))
    hor = dist(left, right)
    ver = dist(top, bottom)
    if hor == 0:
        return 0.0
    return ver / hor
def mouth_width(landmarks, frame_w, frame_h):
    left = (int(landmarks[MOUTH_OUTER[0]][0]*frame_w), int(landmarks[MOUTH_OUTER[0]][1]*frame_h))
    right = (int(landmarks[MOUTH_OUTER[1]][0]*frame_w), int(landmarks[MOUTH_OUTER[1]][1]*frame_h))
    return dist(left, right)
def brow_height(landmarks, brow_idx, eye_idx, frame_w, frame_h):
    brow = (int(landmarks[brow_idx][0]*frame_w), int(landmarks[brow_idx][1]*frame_h))
    eye = (int(landmarks[eye_idx][0]*frame_w), int(landmarks[eye_idx][1]*frame_h))
    return abs(brow[1] - eye[1])  
def draw_landmarks_on_frame(frame, landmark_list, color=(0,255,0)):
    h, w = frame.shape[:2]
    for (x,y) in landmark_list:
        px = int(x * w)
        py = int(y * h)
        cv2.circle(frame, (px, py), 2, color, -1)
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                             refine_landmarks=True,
                             min_detection_confidence=0.56,
                             min_tracking_confidence=0.56)

screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam. Check camera availability and permissions.")
def collect_samples(prompt_text, seconds):
    samples_ear = []
    samples_mar = []
    samples_nosey = []
    samples_mouthw = []
    samples_brow_h = []  
    t0 = time.time()
    while time.time() - t0 < seconds:
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            landmarks = [(p.x, p.y) for p in lm]
            ear_l = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            ear_r = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            ear = (ear_l + ear_r) / 2.0
            mar = mouth_aspect_ratio(landmarks, w, h)
            nosey = landmarks[NOSE_TIP][1]
            mw = mouth_width(landmarks, w, h)
            try:
                bh_l = brow_height(landmarks, LEFT_EYEBROW_POINT, LEFT_EYE_REF, w, h)
                bh_r = brow_height(landmarks, RIGHT_EYEBROW_POINT, RIGHT_EYE_REF, w, h)
                bh = (bh_l + bh_r) / 2.0
            except Exception:
                bh = 0.0
            samples_ear.append(ear)
            samples_mar.append(mar)
            samples_nosey.append(nosey)
            samples_mouthw.append(mw)
            samples_brow_h.append(bh)
            draw_landmarks_on_frame(frame, landmarks)
        cv2.rectangle(frame, (0, frame.shape[0]-110), (frame.shape[1], frame.shape[0]), (0,0,0), -1)
        cv2.putText(frame, prompt_text, (10, frame.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)
        cv2.putText(frame, "Press ESC to abort calibration", (10, frame.shape[0]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            events.append((time.time(), "ESC_ABORT_CALIB"))
            cap.release()
            cv2.destroyAllWindows()
            face_mesh.close()
            raise SystemExit("Calibration aborted by ESC.")
    return samples_ear, samples_mar, samples_nosey, samples_mouthw, samples_brow_h
print("Calibration: follow on-screen prompts (longer calibration enabled).")
open_ear_samples, open_mar_samples, open_nose, open_mouthw, open_brow = collect_samples("Calib 1/5: Keep BOTH EYES OPEN (neutral face)", CALIB_TIME)
closed_ear_samples, closed_mar_samples, closed_nose, closed_mouthw, closed_brow = collect_samples("Calib 2/5: Blink a few times (closed-eye samples)", CALIB_TIME)
mouth_open_ear_samples, mouth_open_mar_samples, mouth_open_nose, mouth_open_mouthw, mouth_open_brow = collect_samples("Calib 3/5: Open MOUTH WIDE and hold", CALIB_TIME)
neutral_ear_samples, neutral_mar_samples, neutral_nose_samples, neutral_mouthw, neutral_brow = collect_samples("Calib 4/5: Face NEUTRAL (head baseline & neutral smile)", CALIB_TIME)
_, _, _, _, eyebrow_samples = collect_samples("Calib 5/5: Relax face (hold neutral eyebrows)", CALIB_TIME)
cv2.destroyWindow("Calibration")
open_ear = np.median(open_ear_samples) if open_ear_samples else 0.28
closed_ear = np.median(closed_ear_samples) if closed_ear_samples else open_ear * 0.6
mouth_neutral = np.median(open_mar_samples) if open_mar_samples else 0.15
mouth_wide = np.median(mouth_open_mar_samples) if mouth_open_mar_samples else mouth_neutral * 2.5
neutral_nose = np.median(neutral_nose_samples) if neutral_nose_samples else 0.5
neutral_mouth_width = np.median(neutral_mouthw) if neutral_mouthw else 40.0
neutral_brow_height = np.median(eyebrow_samples) if eyebrow_samples else 20.0
EYEBROW_THRESH = neutral_brow_height * EYEBROW_FACTOR
EAR_THRESH = (open_ear + closed_ear) / 2.0
MAR_CLICK_THRESH = mouth_neutral + (mouth_wide - mouth_neutral) * 0.35
MAR_DRAG_THRESH = mouth_neutral + (mouth_wide - mouth_neutral) * 0.75
events.append((time.time(), f"CALIB_DONE EAR={EAR_THRESH:.3f} MAR_CLICK={MAR_CLICK_THRESH:.3f} MAR_DRAG={MAR_DRAG_THRESH:.3f} NEUTRAL_NOSE={neutral_nose:.3f} EYEBROW={EYEBROW_THRESH:.2f}"))
print(f"Calibration done. EAR={EAR_THRESH:.3f} MAR_CLICK={MAR_CLICK_THRESH:.3f} MAR_DRAG={MAR_DRAG_THRESH:.3f} NEUTRAL_NOSE={neutral_nose:.3f} EYEBROW_THRESH={EYEBROW_THRESH:.2f}")
SEG_TIME = 4.0
def practice_segment(prompt_text, seconds):
    seg_times = []
    seg_ear = []
    seg_mar = []
    seg_nose = []
    seg_mouthw = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        ear_val = 0.0
        mar_val = 0.0
        nose_y = 0.0
        mw = 0.0
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            landmarks = [(p.x, p.y) for p in lm]
            ear_val = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            ear_val = (ear_val + eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)) / 2.0
            mar_val = mouth_aspect_ratio(landmarks, w, h)
            nose_y = landmarks[NOSE_TIP][1]
            mw = mouth_width(landmarks, w, h)
            draw_landmarks_on_frame(frame, landmarks)
            cv2.circle(frame, (int(landmarks[NOSE_TIP][0]*w), int(landmarks[NOSE_TIP][1]*h)), 4, (0,255,0), -1)
        seg_times.append(time.time())
        seg_ear.append(ear_val)
        seg_mar.append(mar_val)
        seg_nose.append(nose_y)
        seg_mouthw.append(mw)
        times.append(time.time())
        ears.append(ear_val)
        mars.append(mar_val)
        nose_ys.append(nose_y)
        cv2.rectangle(frame, (0, frame.shape[0]-110), (frame.shape[1], frame.shape[0]), (0,0,0), -1)
        cv2.putText(frame, prompt_text, (10, frame.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)
        cv2.putText(frame, f"EAR:{ear_val:.3f} MAR:{mar_val:.3f} NOSE:{nose_y:.3f} MW:{mw:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,255,180), 1)
        cv2.putText(frame, f"Press ESC to abort", (10,frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.imshow("Practice", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            events.append((time.time(), "ESC_ABORT_PRACTICE"))
            cap.release()
            cv2.destroyAllWindows()
            face_mesh.close()
            raise SystemExit("Practice aborted by ESC.")
    return {"times": seg_times, "ear": seg_ear, "mar": seg_mar, "nose": seg_nose, "mouthw": seg_mouthw}
seg_single = practice_segment("Practice 1/4: Single blink (LEFT CLICK) - blink once", SEG_TIME)
seg_double = practice_segment("Practice 2/4: Double blink (RIGHT CLICK) - double blink", SEG_TIME)
seg_triple = practice_segment("Practice 3/4: Triple blink (DOUBLE-CLICK) - blink 3 times", SEG_TIME)
seg_scroll = practice_segment("Practice 4/4: HEAD SCROLL (look UP for up, DOWN for down)", SEG_TIME)
cv2.destroyWindow("Practice")
def rel_times(ts_list):
    if not ts_list:
        return []
    t0 = ts_list[0]
    return [t - t0 for t in ts_list]

plt.figure(figsize=(11,8))
plt.subplot(2,2,1)
rt = rel_times(seg_single.get('times', []))
plt.plot(rt, seg_single.get('ear', []), label='EAR')
plt.title("Single Blink (LEFT CLICK)")
plt.xlabel("Time (s)")
plt.ylabel("EAR")
plt.grid(True)
plt.subplot(2,2,2)
rt = rel_times(seg_double.get('times', []))
plt.plot(rt, seg_double.get('ear', []), label='EAR')
plt.title("Double Blink (RIGHT CLICK)")
plt.xlabel("Time (s)")
plt.ylabel("EAR")
plt.grid(True)
plt.subplot(2,2,3)
rt = rel_times(seg_triple.get('times', []))
plt.plot(rt, seg_triple.get('ear', []), label='EAR')
plt.title("Triple Blink (DOUBLE-CLICK)")
plt.xlabel("Time (s)")
plt.ylabel("EAR")
plt.grid(True)
plt.subplot(2,2,4)
rt = rel_times(seg_scroll.get('times', []))
plt.plot(rt, seg_scroll.get('nose', []), label='nose_y')
plt.axhline(neutral_nose, color='r', linestyle='--', label='neutral nose')
plt.title("Head Scroll Mode (Nose Y)")
plt.xlabel("Time (s)")
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.show()   
print("Graph closed. Live control will start after READY_DELAY.")
time.sleep(READY_DELAY)
blink_state = False
blink_start_time = 0.0
group_start_time = None
blink_count = 0
dragging = False
ema_x, ema_y = screen_w//2, screen_h//2
scroll_state = "neutral"  
eyebrow_start = None
eyebrow_active = False
paused = False
events.append((time.time(), "LIVE_STARTED"))
print("Live control started. Press ESC or raise BOTH EYEBROWS (hold) to quit. Press 'p' to pause/resume.")
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    ear_val = 0.0
    mar_val = 0.0
    nose_y = neutral_nose
    mouthw = 0.0
    brow_h = 0.0
    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        landmarks = [(p.x, p.y) for p in lm]
        ear_l = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
        ear_r = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        ear_val = (ear_l + ear_r) / 2.0
        mar_val = mouth_aspect_ratio(landmarks, w, h)
        nose_y = landmarks[NOSE_TIP][1]
        mouthw = mouth_width(landmarks, w, h)
        try:
            bh_l = brow_height(landmarks, LEFT_EYEBROW_POINT, LEFT_EYE_REF, w, h)
            bh_r = brow_height(landmarks, RIGHT_EYEBROW_POINT, RIGHT_EYE_REF, w, h)
            brow_h = (bh_l + bh_r) / 2.0
        except Exception:
            brow_h = 0.0
        draw_landmarks_on_frame(frame, landmarks)
        nose_x = landmarks[NOSE_TIP][0]
        target_x = int(nose_x * screen_w)
        target_y = int(nose_y * screen_h)
        ema_x = (1 - SMOOTHING_ALPHA) * ema_x + SMOOTHING_ALPHA * target_x
        ema_y = (1 - SMOOTHING_ALPHA) * ema_y + SMOOTHING_ALPHA * target_y
        center_x, center_y = screen_w/2, screen_h/2
        offset_x = (ema_x - center_x) * SENSITIVITY
        offset_y = (ema_y - center_y) * SENSITIVITY
        move_x = int(center_x + offset_x)
        move_y = int(center_y + offset_y)
        move_x = max(0, min(screen_w-1, move_x))
        move_y = max(0, min(screen_h-1, move_y))
        if not paused:
            try:
                pyautogui.moveTo(move_x, move_y, duration=0.01)
            except Exception:
                pass
        now = time.time()
        if brow_h > EYEBROW_THRESH:
            if not eyebrow_active:
                eyebrow_active = True
                eyebrow_start = now
                cv2.putText(frame, "EYEBROWS UP: Exiting in 1.2s...", (10, frame.shape[0]-140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)
            else:
                if now - eyebrow_start >= EYEBROW_HOLD:
                    events.append((now, "EYEBROW_EXIT"))
                    print("Eyebrows held up - exiting.")
                    cv2.putText(frame, "Exiting...", (10, frame.shape[0]-140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                    cv2.imshow("Live Control (ESC to quit)", frame)
                    cv2.waitKey(300)
                    break
        else:
            eyebrow_active = False
            eyebrow_start = None
        if mar_val > MAR_DRAG_THRESH and not paused:
            if not dragging:
                dragging = True
                try:
                    pyautogui.mouseDown(button='left')
                except Exception:
                    pass
                events.append((now, "MOUTH_WIDE -> DRAG_START"))
                print("MOUTH_WIDE -> DRAG_START")
        else:
            if dragging and mar_val < MAR_CLICK_THRESH:
                dragging = False
                try:
                    pyautogui.mouseUp(button='left')
                except Exception:
                    pass
                events.append((now, "MOUTH_CLOSE -> DRAG_END"))
                print("MOUTH_CLOSE -> DRAG_END")
        if not dragging and not paused:
            is_closed = ear_val < EAR_THRESH
            if is_closed and not blink_state:
                blink_state = True
                blink_start_time = now
            if (not is_closed) and blink_state:
                blink_state = False
                if group_start_time is None:
                    group_start_time = now
                    blink_count = 1
                else:
                    if now - group_start_time <= BLINK_WINDOW:
                        blink_count += 1
                    else:
                        group_start_time = now
                        blink_count = 1
                events.append((now, f"BLINK_DETECTED cnt={blink_count}"))
            if group_start_time is not None and (now - group_start_time) > BLINK_WINDOW:
                if blink_count == 1:
                    try:
                        pyautogui.click(button='left')
                    except Exception:
                        pass
                    events.append((now, "SINGLE_BLINK -> LEFT_CLICK"))
                    print("SINGLE_BLINK -> LEFT_CLICK")
                elif blink_count == 2:
                    try:
                        pyautogui.click(button='right')
                    except Exception:
                        pass
                    events.append((now, "DOUBLE_BLINK -> RIGHT_CLICK"))
                    print("DOUBLE_BLINK -> RIGHT_CLICK")
                elif blink_count == 3:
                    try:
                        pyautogui.doubleClick()
                    except Exception:
                        pass
                    events.append((now, "TRIPLE_BLINK -> DOUBLE_CLICK"))
                    print("TRIPLE_BLINK -> DOUBLE_CLICK")
                else:
                    events.append((now, f"IGNORED_BLINKS cnt={blink_count}"))
                    print(f"IGNORED_BLINKS cnt={blink_count}")
                group_start_time = None
                blink_count = 0
        delta = neutral_nose - nose_y
        if not paused:
            if delta > SCROLL_TRIGGER_THRESHOLD and scroll_state == "neutral":
                try:
                    pyautogui.scroll(SCROLL_STEP_AMOUNT)
                except Exception:
                    pass
                events.append((now, "HEAD_UP -> SCROLL_UP"))
                print("HEAD_UP -> SCROLL_UP")
                scroll_state = "up_triggered"
            elif delta < -SCROLL_TRIGGER_THRESHOLD and scroll_state == "neutral":
                try:
                    pyautogui.scroll(-SCROLL_STEP_AMOUNT)
                except Exception:
                    pass
                events.append((now, "HEAD_DOWN -> SCROLL_DOWN"))
                print("HEAD_DOWN -> SCROLL_DOWN")
                scroll_state = "down_triggered"
            elif abs(delta) < SCROLL_NEUTRAL_MARGIN and scroll_state != "neutral":
                scroll_state = "neutral"
        cv2.putText(frame, f"EAR:{ear_val:.3f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"MAR:{mar_val:.3f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,180,0), 2)
        cv2.putText(frame, f"NOSEY:{nose_y:.3f}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,255,180), 1)
        cv2.putText(frame, f"MW:{mouthw:.1f}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,255), 1)
        cv2.putText(frame, f"BROW_H:{brow_h:.1f}", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,255), 1)
        cv2.putText(frame, f"Drag:{dragging}", (10,180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,255), 1)
        cv2.putText(frame, f"Paused:{paused}", (10,210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,255), 1)
    times.append(time.time())
    ears.append(ear_val)
    mars.append(mar_val)
    nose_ys.append(nose_y)
    cv2.imshow("Live Control (ESC to quit / raise eyebrows to quit)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        events.append((time.time(), "EXIT_ESC"))
        break
    if key == ord('p'):
        paused = not paused
        events.append((time.time(), f"PAUSED={paused}"))
        print("Paused" if paused else "Resumed")
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
def rel_times(ts_list):
    if not ts_list:
        return []
    t0 = ts_list[0]
    return [t - t0 for t in ts_list]
plt.figure(figsize=(11,8))
plt.subplot(2,2,1)
rt = rel_times(seg_single.get('times', []))
plt.plot(rt, seg_single.get('ear', []), label='EAR')
plt.title("Single Blink (LEFT CLICK)")
plt.xlabel("Time (s)")
plt.ylabel("EAR")
plt.grid(True)
plt.subplot(2,2,2)
rt = rel_times(seg_double.get('times', []))
plt.plot(rt, seg_double.get('ear', []), label='EAR')
plt.title("Double Blink (RIGHT CLICK)")
plt.xlabel("Time (s)")
plt.ylabel("EAR")
plt.grid(True)
plt.subplot(2,2,3)
rt = rel_times(seg_triple.get('times', []))
plt.plot(rt, seg_triple.get('ear', []), label='EAR')
plt.title("Triple Blink (DOUBLE-CLICK)")
plt.xlabel("Time (s)")
plt.ylabel("EAR")
plt.grid(True)
plt.subplot(2,2,4)
rt = rel_times(seg_scroll.get('times', []))
plt.plot(rt, seg_scroll.get('nose', []), label='nose_y')
plt.axhline(neutral_nose, color='r', linestyle='--', label='neutral nose')
plt.title("Head Scroll Mode (Nose Y)")
plt.xlabel("Time (s)")
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.show()
print("\nEvent Log:")
for ts, ev in events:
    print(f"{time.strftime('%H:%M:%S', time.localtime(ts))} - {ev}")
print("\nProgram ended. Close the plot window to fully exit.")