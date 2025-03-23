import cv2
import numpy as np
import os

project_dir = os.path.dirname(os.path.realpath(__file__))

config_path = os.path.join(project_dir, "yolov4-tiny.cfg")
weights_path = os.path.join(project_dir, "yolov4-tiny_best.weights")
names_path = os.path.join(project_dir, "obj.names")

with open(names_path, "r") as f:
    labels = f.read().strip().split("\n")

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
if net.empty():
    print("ERROR: Model files could not be uploaded!")
    exit()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: The camera could not be turned on!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to capture frame from the camera.")
        break

    (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    layer_outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    confidence_threshold = 0.5
    nms_threshold = 0.4

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]
            color = (206, 183, 255)
            text = f"({labels[class_ids[i]].upper()} %{int(confidences[i] * 100)})"
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)

    cv2.imshow("DETECTION", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()