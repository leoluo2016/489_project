import cv2
import numpy as np
import matplotlib.pyplot as plt

video_path = "489_example.mp4"  # or MOV, whichever you have
template_path = "ball_template.png"

# test comment for github commit part 2
cap = cv2.VideoCapture(video_path)

template = cv2.imread(template_path, 0)
tw, th = template.shape[::-1]

scan_frames = 100
match_threshold = 0.98
launch_delay = 100        # HARD DELAY (frames before launch detection allowed)

candidates = []

print("Scanning early frames for ball candidates...")

# -------- TEMPLATE SCAN --------
for i in range(scan_frames):

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

    locations = np.where(result >= match_threshold)

    for pt in zip(*locations[::-1]):

        cx = int(pt[0] + tw//2)
        cy = int(pt[1] + th//2)

        candidates.append((cx, cy))

print(f'Candidate positions: {len(candidates)} at {candidates}')

# rewind video to start
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

ret, prev_frame = cap.read()

if not ret:
    raise RuntimeError("Failed to read first frame")

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# ---- OUTPUT VIDEO WRITER ----
fps = cap.get(cv2.CAP_PROP_FPS) / 6  # number indicates slow down output for better visualization, adjust as needed
h, w = prev_frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter("tracked_output.mp4", fourcc, fps, (w, h))


kernel = np.ones((3,3), np.uint8)       # for dilating motion areas, modify as needed

ball_center = None
ball_detected = False

trajectory = []
flight_points = []
frame_count = 0

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------- VISUALIZE CANDIDATES --------
    for cand in candidates:
        cv2.circle(frame, cand, 4, (0,255,255), -1)
    # -------- LAUNCH DETECTION --------
    if not ball_detected and frame_count > launch_delay:

        for cand in candidates:

            cx, cy = cand

            x0 = cx - tw//2
            y0 = cy - th//2
            x1 = cx + tw//2
            y1 = cy + th//2

            h, w = gray.shape

            if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                continue

            patch = gray[y0:y1, x0:x1]

            res = cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)

            similarity = res[0][0]

            # ball disappeared
            if similarity < 0.2:        # this threshold may need tuning based on video quality and template match strength

                ball_center = (cx, cy)
                ball_detected = True
                print("Ball launch detected")

                break

    # -------- MOTION TRACKING --------
    if ball_detected:
        diff = cv2.absdiff(gray, prev_gray)
        _, motion = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)     # this threshold may need tuning based on video quality and lighting conditions

        motion = cv2.dilate(motion, kernel, iterations=1)       # iterations controls how much we expand motion areas, modify as needed
        cv2.imshow("Video", motion)     # visualize motion areas for debugging, can be removed in final version

        contours, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_velocity = 0

        x0, y0 = ball_center

        angle = np.radians(10)
        height = gray.shape[0]

        for y in range(y0, 0, -20):

            dy = y0 - y
            fan_half_width = dy * np.tan(angle)

            left = int(x0 - fan_half_width)
            right = int(x0 + fan_half_width)

            cv2.line(frame, (left, y), (right, y), (255,0,0), 1)

        for c in contours:

            area = cv2.contourArea(c)

            if area < 8 or area > 200:
                continue

            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if circularity < 0.4:
                continue

            x, y, w, h = cv2.boundingRect(c)

            cx = x + w//2
            cy = y + h//2

            # ignore motion below launch point
            if cy > y0:
                continue

            # compute fan width
            dy = y0 - cy
            fan_half_width = dy * np.tan(angle)

            left_bound  = x0 - fan_half_width
            right_bound = x0 + fan_half_width

            # reject detections outside fan
            if cx < left_bound or cx > right_bound:
                continue

            dx = cx - ball_center[0]
            dy = cy - ball_center[1]

            velocity = np.sqrt(dx*dx + dy*dy)

            if velocity < 12:
                continue

            if velocity > 350:
                continue

            if velocity > best_velocity:
                best_velocity = velocity
                best = (cx, cy, x, y, w, h)

        if best is not None:

            cx, cy, x, y, w, h = best

            ball_center = (cx, cy)

            trajectory.append(ball_center)

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.circle(frame, (cx,cy), 4, (0,0,255), -1)
            flight_points.append((cx, cy))
            print(f'Frame:{frame_count}, {(cx, cy)}')  # print ball flight points 
    prev_gray = gray
    out.write(frame)
    cv2.imshow("Video", frame)  # visualize tracking, can be removed in final version
    key = cv2.waitKey(25)
    if key == ord('q'):
        break

    if cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE) < 1:
        break
print(flight_points)
cap.release()
out.release()
cv2.destroyAllWindows()