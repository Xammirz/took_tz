import cv2
import tensorflow as tf
import numpy as np
def camera_detection():
    CONFIDENCE = 0.5
    SCORE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    config_path = "yolov3.cfg"
    weights_path = "yolov3.weights"
    font_scale = 0
    thickness = 0
    LABELS = open("coco.names").read().strip().split("\n")
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    ln = net.getLayerNames()
    try:
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except IndexError:
        # в случае, если getUnconnectedOutLayers() возвращает массив 1D, когда CUDA недоступен
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    cap = cv2.VideoCapture(0)

    while True:
        #кадры с камеры
        _, image = cap.read()
        #меняем расширения
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        
        net.setInput(blob)
        layer_outputs = net.forward(ln)
        
        boxes, confidences, class_ids = [], [], []

        # цикл по каждому из выходных данных слоя
        for output in layer_outputs:
            # цикл по каждому из обнаружений объекта
            for detection in output:
                
                # extract the class id (label) and confidence (as a probability) of
                # the current object detection
                scores = detection[5:]
                
                class_id = np.argmax(scores)
                
                confidence = scores[class_id]
                
                # discard weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > CONFIDENCE:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # perform the non maximum suppression given the scores defined before
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

        font_scale = 1
        thickness = 1

        # ensure at least one detection exists
        if len(idxs) > 0:
            
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]
                # draw a bounding box rectangle and label on the image

                color = [int(c) for c in COLORS[class_ids[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
                text = f"{LABELS[class_ids[i]]}: {confidences[i]:.2f}"
                if h > 270 and w > 270 and h < 350 and w < 390:
                    if x < 130 and x > 0:
                        x = 330
                        y = 100
                        print(f'слева X: {x} Y: {y}')
                    if x > 130 and x < 200:
                        x = 400
                        y = 100
                        print(f'посередине X: {x} Y: {y}')

                    if x > 200 and x < 300:
                        x = 1
                        y = 100
                        print(f'справа X: {x} Y: {y}')
                else:
                    print(f'Height: {h} Widht: {w}\nизображение слишком близко либо слишком далеко')
                # calculate text width & height to draw the transparent boxes as background of the text
                (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
                text_offset_x = x
                text_offset_y = y - 5
                box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                overlay = image.copy()
                cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
                # add opacity (transparency to the box)
                image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
                # now put the text (label: confidence %)
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

        cv2.imshow("image", image)
        if ord("q") == cv2.waitKey(10):
            break

    cap.release()
    cv2.destroyAllWindows()
def my_model():
    print(tf.__version__)
camera_detection()