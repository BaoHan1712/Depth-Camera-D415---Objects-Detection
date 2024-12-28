import cv2
from ultralytics import YOLO
import math
import time
import numpy as np
import pyrealsense2 as rs

# Khởi tạo pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color,1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# Lấy thông tin về các chế độ được hỗ trợ
pipeline_profile = pipeline.start(config)

# Tạo align objec depth 
align_to = rs.stream.color
align = rs.align(align_to)

model = YOLO("cnn.pt")

prev_frame_time = 0
new_frame_time = 0

try:
    while True:
        # Đợi frame mới
        frames = pipeline.wait_for_frames()
        
        # Căn chỉnh frame
        aligned_frames = align.process(frames)
        
        # Lấy frame màu và depth đã căn chỉnh
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue
        # Chuyển đổi frame thành numpy array
        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        detections = np.empty((0, 5))

        new_frame_time = cv2.getTickCount()
        frame = cv2.resize(frame, (1080, 720))

        results = model.predict(source=frame, imgsz=640, conf = 0.55, verbose=False)
        
        fps = cv2.getTickFrequency() / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps_text = f'FPS: {int(fps)}'
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imshow('Object Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()