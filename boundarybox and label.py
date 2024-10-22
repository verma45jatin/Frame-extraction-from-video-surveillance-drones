import cv2
import numpy as np
import os
from object_detection import ObjectDetection
import math

# Initialize Object Detection
od = ObjectDetection()

cap = cv2.VideoCapture("E:\Design Credit Project\los_angeles.mp4")

# Create a directory to store annotation files and images
os.makedirs("annotations", exist_ok=True)
os.makedirs("output_images", exist_ok=True)

# Initialize count
count = 0
center_points_prev_frame = []

tracking_objects = {}
track_id = 0

while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    # Point current frame
    center_points_cur_frame = []

    # Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)

    # Open annotation file for the current frame
    annotation_file = open(f"annotations/frame_{count}.txt", "w")

    for i, box in enumerate(boxes):
        (x, y, w, h) = box
        z = x + w
        w = y + h
        cx = int((x + z) / 2)
        cy = int((y + w) / 2)
        center_points_cur_frame.append((cx, cy))

        # Get the class label for the detected object
        label = od.get_label(class_ids[i])
        label_x = x
        label_y = y - 10 if y - 10 > 10 else y + 10
        # Save the bounding box coordinates and the label to the annotation file
        annotation_file.write(f"{label} {x , y ,z, w ,label_x, label_y,0,0 } \n")

        # Draw the bounding box
        cv2.rectangle(frame, (x, y), (z, w), (0, 255, 0), 2)

        # Draw the label on the frame
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Close the annotation file
    annotation_file.close()

    # Save the image with the bounding box drawn on it
    cv2.imwrite(f"output_images/frame_{count}.jpg", frame)

    # Only at the beginning do we compare previous and current frames
    if count <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 20:
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:
        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                # Update IDs position
                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue

            # Remove IDs lost
            if not object_exists:
                tracking_objects.pop(object_id)

        # Add new IDs found
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1

    # Make a copy of the points
    center_points_prev_frame = center_points_cur_frame.copy()

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
