from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import tempfile
import cv2
import base64
import numpy as np
from ultralytics import YOLO


# Load YOLOv8 model once
yolo_model = YOLO("yolov8s.pt")

class VideoDetectView(APIView):
    def post(self, request):
        try:
            if 'file' not in request.FILES:
                return Response({"error": "No video uploaded"}, status=status.HTTP_400_BAD_REQUEST)

            # Save uploaded video temporarily
            video_file = request.FILES['file']
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            for chunk in video_file.chunks():
                temp_file.write(chunk)
            temp_file.close()

            # Open video and read the first frame
            cap = cv2.VideoCapture(temp_file.name)
            success, frame = cap.read()
            cap.release()

            if not success or frame is None:
                return Response({"error": "Could not read first frame"}, status=status.HTTP_400_BAD_REQUEST)

            # Run YOLO detection on first frame
            results = yolo_model.predict(frame, conf=0.5, verbose=False)

            detected_counts = {}
            annotated = frame.copy()

            # Loop through detections in first frame
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = yolo_model.names[cls_id].lower()

                    if conf < 0.5:
                        continue

                    # Count all detected labels
                    detected_counts[label] = detected_counts.get(label, 0) + 1

                    # Draw detection box and label
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"{label} {conf:.2f}", (x1, max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Convert annotated frame to Base64
            _, buffer = cv2.imencode('.jpg', annotated)
            annotated_base64 = base64.b64encode(buffer).decode('utf-8')

            # Build readable summary
            summary = [f"{count} {label}{'s' if count != 1 else ''}" for label, count in detected_counts.items()]
            summary_text = ", ".join(summary) if summary else "No objects detected"

            return Response({
                "detected_summary": summary_text,
                "detected_items": detected_counts,
                # "annotated_frame_base64_jpg": annotated_base64
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
