import cv2
import torch
import pathlib
import numpy as np

pathlib.PosixPath = pathlib.WindowsPath


model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_oil.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)


cap = cv2.VideoCapture("test_video")


polygon_points_shelf_1 = np.array([[60, 90], [78, 176], [1173, 218], [1195, 156]], np.int32)
polygon_points_shelf_2 = np.array([[96, 203], [133, 266], [1183, 305], [1209, 241]], np.int32)
polygon_points_shelf_3 = np.array([[140, 268], [170, 321], [1136, 352], [1160, 312]], np.int32)
polygon_points_shelf_4 = np.array([[176, 319], [207, 372], [1106, 391], [1137, 363]], np.int32)
polygon_points_shelf_5 = np.array([[215, 381], [243, 429], [1061, 444], [1092, 404]], np.int32)
polygon_points_shelf_1 = polygon_points_shelf_1.reshape((-1, 1, 2))
polygon_points_shelf_2 = polygon_points_shelf_2.reshape((-1, 1, 2))
polygon_points_shelf_3 = polygon_points_shelf_3.reshape((-1, 1, 2))
polygon_points_shelf_4 = polygon_points_shelf_4.reshape((-1, 1, 2))
polygon_points_shelf_5 = polygon_points_shelf_5.reshape((-1, 1, 2))



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))

    cv2.polylines(frame, [polygon_points_shelf_1], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.polylines(frame, [polygon_points_shelf_2], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.polylines(frame, [polygon_points_shelf_3], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.polylines(frame, [polygon_points_shelf_4], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.polylines(frame, [polygon_points_shelf_5], isClosed=True, color=(255, 0, 0), thickness=2)


    results = model(frame)
    bbox = results.xyxy[0].cpu().numpy()
    labels = results.names


    detected_labels_shelf_1 = []
    detected_labels_shelf_2 = []
    detected_labels_shelf_3 = []
    detected_labels_shelf_4 = []
    detected_labels_shelf_5 = []

    for i in bbox:
        confidence = i[4]
        if confidence > 0.5:
            x1, y1, x2, y2, conf, cls = i
            label = labels[int(cls)]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            if cv2.pointPolygonTest(polygon_points_raf_1, (center_x, center_y), False) >= 0:
                detected_labels_shelf_1.append(label)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)

            elif cv2.pointPolygonTest(polygon_points_raf_2, (center_x, center_y), False) >= 0:
                detected_labels_shelf_2.append(label)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)

            elif cv2.pointPolygonTest(polygon_points_raf_3, (center_x, center_y), False) >= 0:
                detected_labels_shelf_3.append(label)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)

            elif cv2.pointPolygonTest(polygon_points_raf_4, (center_x, center_y), False) >= 0:
                detected_labels_shelf_4.append(label)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)

            elif cv2.pointPolygonTest(polygon_points_raf_5, (center_x, center_y), False) >= 0:
                detected_labels_shelf_5.append(label)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)


    if "bos_raf" in detected_labels_shelf_1 and len(detected_labels_shelf_1) >= 1:
        cv2.putText(frame, "Shelf 1 Empty", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Shelf 1 Full", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if "bos_raf" in detected_labels_shelf_2 and len(detected_labels_shelf_2) >= 1:
        cv2.putText(frame, "Shelf 2 Empty", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Shelf 2 Full", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if "bos_raf" in detected_labels_shelf_3 and len(detected_labels_shelf_3) >= 1:
        cv2.putText(frame, "Shelf 3 Empty", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Shelf 3 Full", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if "bos_raf" in detected_labels_shelf_4 and len(detected_labels_shelf_4) >= 1:
        cv2.putText(frame, "Shelf 4 Empty", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Shelf 4 Full", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if "bos_raf" in detected_labels_shelf_5 and len(detected_labels_shelf_5) >= 1:
        cv2.putText(frame, "Shelf 5 Empty", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Shelf 5 Full", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()