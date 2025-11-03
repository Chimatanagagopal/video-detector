from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from ultralytics import YOLO
import cv2
import tempfile
import base64
import time
import os

# 🧠 Render fix: use /tmp as YOLO config directory (since /opt/render/.config is not writable)
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

# ⚙️ Use lightweight YOLO model for Render (smaller, faster)
yolo_model = YOLO("yolov8n.pt")  # 'n' = nano version, much smaller than 's'

class VideoDetectView(APIView):
    def post(self, request):
        start_time = time.time()  # Start timing

        try:
            # ✅ Check if a video file was uploaded
            if 'file' not in request.FILES:
                return Response({"error": "No video uploaded"}, status=status.HTTP_400_BAD_REQUEST)

            # ✅ Save uploaded video temporarily
            video_file = request.FILES['file']
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                for chunk in video_file.chunks():
                    temp_file.write(chunk)
                temp_path = temp_file.name

            # ✅ Capture the first frame
            cap = cv2.VideoCapture(temp_path)
            success, frame = cap.read()
            cap.release()

            # ✅ Clean up temporary file
            os.remove(temp_path)

            if not success or frame is None:
                return Response({"error": "Could not read first frame"}, status=status.HTTP_400_BAD_REQUEST)

            # ✅ Run YOLOv8 detection (Render-safe: disable saving/logging)
            results = yolo_model.predict(frame, conf=0.5, verbose=False, save=False)

            detected_counts = {}
            annotated = frame.copy()

            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < 0.5:
                        continue

                    cls_id = int(box.cls[0])
                    label = yolo_model.names[cls_id].lower()
                    detected_counts[label] = detected_counts.get(label, 0) + 1

                    # Draw detection boxes and labels
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated, f"{label} {conf:.2f}",
                        (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1
                    )

            # ✅ Convert annotated frame to Base64 (optional)
            _, buffer = cv2.imencode('.jpg', annotated)
            annotated_base64 = base64.b64encode(buffer).decode('utf-8')

            # ✅ Prepare summary
            summary = [f"{count} {label}{'s' if count != 1 else ''}" for label, count in detected_counts.items()]
            summary_text = ", ".join(summary) if summary else "No objects detected"

            total_time = round(time.time() - start_time, 4)
            print(f"✅ Detection completed in {total_time} seconds")

            return Response({
                "detected_summary": summary_text,
                "detected_items": detected_counts,
                "processing_time_seconds": total_time,
                # "annotated_frame_base64": annotated_base64  # Uncomment if needed
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
