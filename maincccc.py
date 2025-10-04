import cv2
import torch
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [2, 5, 7]  # car, bus, truck
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# ROI (Region of Interest)
region_top_left = (220, 2)
region_bottom_right = (560, 170)
PARKED_CARS = 6

def count_vehicles(frame):
    frame_resized = cv2.resize(frame, (640, 360))
    x1, y1 = region_top_left
    x2, y2 = region_bottom_right
    roi = frame_resized[y1:y2, x1:x2]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    results = model(roi_rgb)
    detections = results.xyxy[0]

    vehicle_count = 0
    boxes = []

    for *box, conf, cls in detections:
        x1_box, y1_box, x2_box, y2_box = [int(coord.item()) for coord in box]
        area = (x2_box - x1_box) * (y2_box - y1_box)
        if area < 400:
            continue

        abs_box = (
            x1 + x1_box,
            y1 + y1_box,
            x1 + x2_box,
            y1 + y2_box
        )
        boxes.append((abs_box, conf.item()))  # Store box and confidence
        vehicle_count += 1

    adjusted_vehicle_count = max(vehicle_count - PARKED_CARS, 0)
    return adjusted_vehicle_count, boxes, frame_resized

def draw_boxes(boxes, frame):
    for (x1, y1, x2, y2), conf in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Conf: {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.rectangle(frame, region_top_left, region_bottom_right, (0, 255, 255), 2)
    cv2.putText(frame, "Detection Zone", (region_top_left[0], region_top_left[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return frame

def calculate_wait_time(vehicle_count):
    # Heuristic function: wait time increases with vehicle count
    return vehicle_count * 3  # each vehicle adds 3 seconds

def a_star_signal_decision(vehicle_count):
    start_state = 'red'
    goal_state = 'green'

    open_list = [(start_state, 0)]
    visited = set()

    while open_list:
        state, cost = open_list.pop(0)
        if state == goal_state:
            return 'green' if vehicle_count > 2 else 'red'

        if state not in visited:
            visited.add(state)
            wait_time = calculate_wait_time(vehicle_count)
            open_list.append(('green', cost + wait_time))

    return 'red'


# Main code
cap = cv2.VideoCapture("traffic.mp4")
signal_state = 'red'  # Initial signal state
waiting_cars = 0
paused = False  # Initially not paused

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        vehicle_count, boxes, frame_resized = count_vehicles(frame)
        frame_resized = draw_boxes(boxes, frame_resized)
        
        # Update waiting cars
        waiting_cars = vehicle_count

        # Signal decision (Green or Red)
        signal_state = a_star_signal_decision(waiting_cars)

        # Draw the signal circles (Green or Red)
        signal_color = (0, 255, 0) if signal_state == 'green' else (0, 0, 255)
        cv2.circle(frame_resized, (frame_resized.shape[1] - 50, 320), 30, signal_color, -1)
        cv2.putText(frame_resized, f"Signal: {signal_state.upper()}", (frame_resized.shape[1] - 200, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, signal_color, 2)

        # Show the number of waiting cars
        cv2.putText(frame_resized, f"Waiting Cars: {waiting_cars+1}", (frame_resized.shape[1] - 200, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the frame with UI
    cv2.imshow("Adaptive Traffic Management", frame_resized)

    # Check for user input to pause/resume
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit on 'q'
        break
    if key == ord(' '):  # Pause/resume on spacebar
        paused = not paused

cap.release()
cv2.destroyAllWindows()
