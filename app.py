import sys
import os
from pathlib import Path
import base64
import time
from datetime import datetime
import subprocess
import threading
import traceback
import numpy as np
from typing import List, Dict, Tuple, Optional
from flask import Flask, request, jsonify, Response
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import torch
from PIL import Image
import io
from collections import defaultdict
ddddddddddd
# 현재 디렉토리 설정
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
yolov9_dir = project_root / 'yolov9'
sys.path.append(str(yolov9_dir))

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.plots import Annotator, colors

# 위험 클래스를 정의합니다.
DANGER_CLASSES = ['drowning', 'floundering', 'slipping']
EXCLUDE_CLASSES = ['swimming']  # 수영 중인 경우는 위험 상황에서 제외

# 위험 지속 시간 설정
DANGER_DURATION_THRESHOLD = 3.0  # 위험 상황을 판단하는 지속 시간
DANGER_RECHECK_INTERVAL = 2.0  # 위험 상태가 유지될 수 있는 최대 간격 (초 단위)

class AlertManager:
    def __init__(self, memory_duration=5.0, alert_threshold=3, reset_duration=3.0):
        self.danger_states = {
            'drowning': {'detections': [], 'last_alert': None, 'consecutive_count': 0},
            'floundering': {'detections': [], 'last_alert': None, 'consecutive_count': 0},
            'slipping': {'detections': [], 'last_alert': None, 'consecutive_count': 0}
        }
        self.memory_duration = memory_duration
        self.alert_threshold = alert_threshold
        self.reset_duration = reset_duration
        self.danger_positions = {}  # 객체별 위치 기록
        self.danger_recheck = defaultdict(lambda: {'last_seen': None, 'count': 0})  # 위치별 위험 지속성 기록

    def _clean_old_detections(self, current_time: float, danger_type: str):
        """오래된 감지 기록 제거"""
        state = self.danger_states[danger_type]
        state['detections'] = [det for det in state['detections'] 
                               if current_time - det <= self.memory_duration]

    def update(self, current_time: float, detections: List[Dict]) -> List[Dict]:
        """감지 상태 업데이트 및 알림 생성"""
        alerts = []
        
        # 현재 프레임에서 감지된 위험 객체의 위치를 추출
        for det in detections:
            if det['cls'] in DANGER_CLASSES and det['cls'] not in EXCLUDE_CLASSES:
                x1, y1, x2, y2 = det['xyxy']
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                position_key = (center_x, center_y)
                
                # 기존에 감지된 위치일 경우, 일시적 중단을 고려하여 지속성 체크
                if position_key in self.danger_recheck:
                    last_seen = self.danger_recheck[position_key]['last_seen']
                    # 일시적으로 감지가 중단된 경우에도 연속적으로 감지된 것으로 처리
                    if current_time - last_seen <= DANGER_RECHECK_INTERVAL:
                        self.danger_recheck[position_key]['count'] += 1
                    else:
                        # 일정 시간 경과 시 카운트 초기화
                        self.danger_recheck[position_key]['count'] = 1
                else:
                    # 새로운 위치로 감지 시작
                    self.danger_recheck[position_key]['count'] = 1

                # 현재 감지 시간 기록
                self.danger_recheck[position_key]['last_seen'] = current_time
                
                # 감지 횟수가 위험 지속 시간 조건을 충족할 경우 위험 상황으로 판단
                if (self.danger_recheck[position_key]['count'] >= self.alert_threshold and 
                    current_time - self.danger_recheck[position_key]['last_seen'] <= DANGER_DURATION_THRESHOLD):
                    
                    alert = {
                        'type': det['cls'],
                        'message': f"{det['cls'].capitalize()} 상황이 {self.alert_threshold}회 이상 감지되었습니다.",
                        'timestamp': current_time,
                        'confidence': 'high' if self.danger_recheck[position_key]['count'] > self.alert_threshold else 'medium'
                    }
                    alerts.append(alert)
                    
                    # 알림 생성 후 위치 및 카운트 초기화
                    self.danger_recheck[position_key]['count'] = 0

        return alerts

    def get_danger_level(self) -> str:
        """전반적인 위험 수준 평가"""
        total_recent_detections = sum(
            len(state['detections']) for state in self.danger_states.values()
        )
        if total_recent_detections >= self.alert_threshold * 2:
            return 'high'
        elif total_recent_detections >= self.alert_threshold:
            return 'medium'
        return 'low'
    
# AlertManager 초기화
alert_manager = AlertManager()

# DetectionTracker 클래스 정의
class DetectionTracker:
    def __init__(self, memory_frames=5, iou_threshold=0.3):
        self.memory_frames = memory_frames
        self.iou_threshold = iou_threshold
        self.tracked_objects = {}

    def iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2

        inter_x1 = max(x1, x1_p)
        inter_y1 = max(y1, y1_p)
        inter_x2 = min(x2, x2_p)
        inter_y2 = min(y2, y2_p)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_p - x1_p) * (y2_p - y1_p)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def update(self, detections, current_time):
        """Update tracked objects based on new detections."""
        new_tracked_objects = {}
        
        for det in detections:
            best_match_id = None
            best_iou = self.iou_threshold
            
            for obj_id, obj in self.tracked_objects.items():
                iou_score = self.iou(det['xyxy'], obj['bbox'])
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_match_id = obj_id
            
            if best_match_id is not None:
                prev_bbox = self.tracked_objects[best_match_id]['bbox']
                new_bbox = det['xyxy']
                averaged_bbox = [
                    (prev_bbox[i] + new_bbox[i]) / 2 for i in range(4)
                ]
                new_tracked_objects[best_match_id] = {
                    'bbox': averaged_bbox,
                    'timestamp': current_time,
                    'cls': det['cls']
                }
            else:
                new_id = len(self.tracked_objects) + 1
                new_tracked_objects[new_id] = {
                    'bbox': det['xyxy'],
                    'timestamp': current_time,
                    'cls': det['cls']
                }

        self.tracked_objects = {
            obj_id: obj for obj_id, obj in new_tracked_objects.items()
            if current_time - obj['timestamp'] <= self.memory_frames
        }

        return list(self.tracked_objects.values())

# LaneManager 클래스 정의
class LaneManager:
    def __init__(self, num_lanes=8, lane_width=80):
        self.num_lanes = num_lanes
        self.lane_width = lane_width  # 각 레인의 너비를 설정 (픽셀 단위)

    def get_lane(self, x):
        lane = int(x / self.lane_width)
        return min(lane, self.num_lanes - 1)

    def assign_lane(self, detection):
        x1, y1, x2, y2 = detection['xyxy']
        center_x = (x1 + x2) / 2
        lane = self.get_lane(center_x)
        return lane

    def adjust_position_to_lane(self, detection):
        lane = self.assign_lane(detection)
        lane_start = lane * self.lane_width
        lane_end = lane_start + self.lane_width

        x1, y1, x2, y2 = detection['xyxy']
        adjusted_x1 = max(x1, lane_start)
        adjusted_x2 = min(x2, lane_end)
        
        detection['xyxy'] = (adjusted_x1, y1, adjusted_x2, y2)
        detection['lane'] = lane

        return detection

    def get_danger_level(self) -> str:
        """전반적인 위험 수준 평가"""
        total_recent_detections = sum(
            len(state['detections']) for state in self.danger_states.values()
        )
        if total_recent_detections >= self.alert_threshold * 2:
            return 'high'
        elif total_recent_detections >= self.alert_threshold:
            return 'medium'
        return 'low'

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# 파일 업로드 설정
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB 제한
TEMP_DIR = 'temp_uploads'
os.makedirs(TEMP_DIR, exist_ok=True)

# 모델 설정
weights_path = 'C:/Users/user/Desktop/yolov9/pool-safety-system/yolov9/yolov9-weights.pt'
device = select_device('')
data = yolov9_dir / 'data/data.yaml'
imgsz = (640, 640)
model = DetectMultiBackend(weights_path, device=device, dnn=False, data=data, fp16=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)
model.warmup(imgsz=(1 if pt else 1, 3, *imgsz))

alert_manager = AlertManager()

def process_frame(frame, tracker, lane_manager):
    """단일 프레임 처리"""
    frame_height, frame_width = frame.shape[:2]
    current_time = time.time()
    
    # 모델 입력 크기를 동적으로 설정
    img_tensor = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)

    # 객체 검출
    with torch.no_grad():
        pred = model(img_tensor)
        if isinstance(pred, list):
            pred = pred[0]
        
        pred = non_max_suppression(
            pred,
            conf_thres=0.3,
            iou_thres=0.3,
            max_det=50
        )

    frame_with_boxes = frame.copy()
    valid_detections = []

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], frame.shape).round()
            
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                class_name = names[int(cls)]
                
                valid_detections.append({
                    'xyxy': (x1, y1, x2, y2),
                    'conf': float(conf),
                    'cls': class_name,
                    'area': (x2 - x1) * (y2 - y1),
                    'timestamp': current_time
                })

    # 결과 시각화
    danger_detected = False
    for det in valid_detections:
        x1, y1, x2, y2 = det['xyxy']
        class_name = det['cls']
        conf = det['conf']
        color = (0, 0, 255) if class_name in DANGER_CLASSES else (0, 255, 0)
        
        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        label = f"{class_name} {conf:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame_with_boxes,
                     (x1, y1 - label_size[1] - 8),
                     (x1 + label_size[0], y1),
                     color, -1)
        cv2.putText(frame_with_boxes,
                   label,
                   (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5,
                   (255, 255, 255),
                   2)

        if class_name in DANGER_CLASSES:
            danger_detected = True

    return frame_with_boxes, danger_detected, valid_detections

# process_video 함수 내에서 frame_with_boxes로 수정
def process_frame(frame, img_tensor, tracker, lane_manager):
    """단일 프레임 처리 및 라벨 그리기"""
    frame_height, frame_width = frame.shape[:2]
    current_time = time.time()

    # 검출 파라미터
    detection_params = {
        'conf_thres': 0.3,
        'iou_thres': 0.3,
        'max_det': 50,
    }

    # 객체 검출
    with torch.no_grad():
        pred = model(img_tensor)
        if isinstance(pred, list):
            pred = pred[0]

        pred = non_max_suppression(
            pred,
            conf_thres=detection_params['conf_thres'],
            iou_thres=detection_params['iou_thres'],
            max_det=detection_params['max_det']
        )

    frame_with_boxes = frame.copy()
    valid_detections = []

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], frame.shape).round()
            
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                class_name = names[int(cls)]
                
                valid_detections.append({
                    'xyxy': (x1, y1, x2, y2),
                    'conf': float(conf),
                    'cls': class_name,
                    'area': (x2 - x1) * (y2 - y1),
                    'timestamp': current_time
                })

                # 라벨 및 박스 그리기
                color = (0, 0, 255) if class_name in DANGER_CLASSES else (0, 255, 0)
                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame_with_boxes, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    danger_detected = any(det['cls'] in DANGER_CLASSES for det in valid_detections)
    return frame_with_boxes, danger_detected, valid_detections


def process_video(temp_path: str, output_path: str) -> Tuple[bool, Optional[str]]:
    try:
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return False, "비디오 파일을 열 수 없습니다."

        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        temp_output_path = output_path.replace('.mp4', '_temp.avi')
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (original_width, original_height))

        if not out.isOpened():
            cap.release()
            return False, "비디오 작성기를 초기화할 수 없습니다."

        tracker = DetectionTracker(memory_frames=5)
        lane_manager = LaneManager(num_lanes=8)
        frame_count = 0
        danger_detected = False

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                resized_frame = cv2.resize(frame, (640, 640))
                img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                if len(img_tensor.shape) == 3:
                    img_tensor = img_tensor.unsqueeze(0)
                img_tensor = torch.nn.functional.interpolate(img_tensor, size=imgsz, mode='bilinear', align_corners=False)
                img_tensor = img_tensor.to(device)

                frame_with_boxes, frame_danger, _ = process_frame(resized_frame, img_tensor, tracker, lane_manager)
                danger_detected = danger_detected or frame_danger

                # 원본 해상도로 다시 조정하여 저장
                output_frame = cv2.resize(frame_with_boxes, (original_width, original_height))
                out.write(output_frame)

                if frame_count % 30 == 0:
                    progress = min((frame_count / total_frames) * 100, 100)
                    socketio.emit('processing_progress', {
                        'progress': progress,
                        'frame': frame_count,
                        'total_frames': total_frames
                    })

        finally:
            cap.release()
            out.release()

        # FFmpeg 변환
        try:
            ffmpeg_path = ('C:/Program Files/ffmpeg-2024-10-31-git-87068b9600-essentials_build/bin/ffmpeg.exe' 
                           if os.name == 'nt' else 'ffmpeg')

            result = subprocess.run([
                ffmpeg_path,
                '-f', 'avi',
                '-i', temp_output_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-y',
                output_path,
                '-report'
            ], check=True, capture_output=True, text=True)

            if result.stderr:
                print("FFmpeg 경고/에러:", result.stderr)

        except subprocess.CalledProcessError as e:
            print(f"FFmpeg 처리 중 오류: {str(e)}")
            print(f"FFmpeg stderr 로그: {e.stderr}")
            return danger_detected, f"FFmpeg 처리 오류: {e.stderr}"
        
        except Exception as e:
            print(f"예외 발생: {str(e)}")
            traceback.print_exc()
            return danger_detected, f"일반 오류: {str(e)}"

        finally:
            if os.path.exists(temp_output_path):
                try:
                    os.remove(temp_output_path)
                except Exception as e:
                    print(f"임시 파일 삭제 중 오류: {str(e)}")

        return danger_detected, None

    except Exception as e:
        print(f"비디오 처리 중 오류 발생: {str(e)}")
        traceback.print_exc()
        return False, str(e)

def process_image(image_data: np.ndarray) -> Tuple[Optional[str], Optional[bool], Optional[str]]:
    """이미지 처리 함수"""
    try:
        tracker = DetectionTracker(memory_frames=1)  # 단일 이미지용
        lane_manager = LaneManager(num_lanes=8)

        img_tensor = torch.from_numpy(image_data.transpose(2, 0, 1)).float() / 255.0
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)
        img_tensor = torch.nn.functional.interpolate(img_tensor, size=imgsz, 
                                                   mode='bilinear', align_corners=False)
        img_tensor = img_tensor.to(device)

        # 이미지 처리
        processed_frame, danger_detected = process_frame(
            image_data, 
            img_tensor,
            tracker,
            lane_manager
        )

        # 이미지를 base64로 변환
        _, buffer = cv2.imencode('.jpg', processed_frame)
        img_str = base64.b64encode(buffer).decode()

        return img_str, danger_detected, None  # 세 개의 값을 반환
    except Exception as e:
        return None, None, str(e)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """파일 업로드 처리 라우트"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        filename = file.filename.lower()
        is_video = filename.endswith(('.mp4', '.avi', '.mov'))
        
        if is_video:
            # 비디오 처리
            temp_path = os.path.join(TEMP_DIR, f'temp_video_{int(time.time())}.mp4')
            output_path = os.path.join(TEMP_DIR, f'output_video_{int(time.time())}.mp4')
            
            try:
                file.save(temp_path)
                danger_detected, error = process_video(temp_path, output_path)
                
                if error:
                    return jsonify({'error': error}), 400
                
                with open(output_path, 'rb') as video_file:
                    video_base64 = base64.b64encode(video_file.read()).decode()
                
                return jsonify({
                    'processed_video': video_base64,
                    'danger_detected': danger_detected,
                    'message': '비디오 처리 완료'
                })
                
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                if os.path.exists(output_path):
                    os.remove(output_path)
        
        else:
            # 이미지 처리
            image_bytes = file.read()
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img_np = np.array(img)
            
            img_str, danger_detected, error = process_image(img_np)
            if error:
                return jsonify({'error': error}), 400

            return jsonify({
                'image': img_str,
                'danger_detected': danger_detected,
                'message': '이미지 처리 완료'
            })
            
    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def cleanup_temp_files():
    """임시 파일 정리 함수"""
    while True:
        for filename in os.listdir(TEMP_DIR):
            filepath = os.path.join(TEMP_DIR, filename)
            if os.path.getctime(filepath) < time.time() - 3600:  # 1시간 이상 된 파일 삭제
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"파일 삭제 중 오류: {str(e)}")
        time.sleep(3600)  # 1시간마다 실행

if __name__ == '__main__':
    threading.Thread(target=cleanup_temp_files, daemon=True).start()
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)