"""
辨識車牌+字元+追蹤+校正
立柱＋rfid+資料庫
"""
import torch
import cv2
import numpy as np
import argparse
import time
import re
from bytetrack.byte_tracker import BYTETracker
from bytetrack.basetrack import BaseTrack
from ultralytics import YOLO
from abc import ABC, abstractmethod
import logging
from datetime import datetime, timedelta
import base64
import threading
import queue
import av
from update_epc_mysql import *
from log_config import setup_logging, get_location

setup_logging(enable_logging=True)
current_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
last_clear_time = time.time()

def array_to_base64(arr):
    _, encoded = cv2.imencode('.png', arr)
    return base64.b64encode(encoded).decode('utf-8')

def resize_to_max_edge(image, max_edge=416):
    height, width = image.shape[:2]
    if height > width:
        new_height = max_edge
        ratio = max_edge / height
        new_width = int(width * ratio)
    else:
        new_width = max_edge
        ratio = max_edge / width
        new_height = int(height * ratio)
    
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

class VideoCapture:
    streaming = True
    def __init__(self, rtsp):
        self.rtsp = rtsp
        self.cap = cv2.VideoCapture(rtsp)
        self.q = queue.Queue()
        self.ret = True
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while VideoCapture.streaming:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    raise Exception("Could not read frame")
                if not self.q.empty():
                    try:
                        self.q.get_nowait()   # discard previous (unprocessed) frame
                    except queue.Empty:
                        pass
                self.ret = ret
                self.q.put(frame)
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)  # wait before retrying
                self.cap.open(self.rtsp)  # reopen the connection
                logging.error('camera connecting failed', exc_info=True)

    def read(self):
        return self.ret, self.q.get()
    
    def release(self):
        self.cap.release() 
    
    def get(self, info):
        return int(self.cap.get(info))
    
    def isOpened(self):
        return self.cap.isOpened() and VideoCapture.streaming

class yolov8Inference(ABC):
    def __init__(self, weights_path, device, class_dict, confidence_threshold):
        # load YOLOv8 model
        self.model = YOLO(weights_path).to(device=device)
        self.class_dict = class_dict
        self.confidence_threshold = confidence_threshold
        self.filtered_detections = None

    @abstractmethod
    def detect(self, img):
        pass

class plateInference(yolov8Inference):
    def __init__(self, weights_path, device, class_dict, confidence_threshold):
        # load YOLOv8 model
        self.model = YOLO(weights_path, task="pose") #.to(device=device)
        self.device = device

        if self.model.task == "pose":
            self.is_pose_model = True
        else:
            self.is_pose_model = False

        self.class_dict = class_dict
        self.confidence_threshold = confidence_threshold
        # 定義車牌的寬和高
        self.plate_width, self.plate_height = 416, 208

        # 定義校正後的四個角點坐標
        self.destination_corners = np.array([[0, 0], [self.plate_width - 1, 0], [self.plate_width - 1, self.plate_height - 1], [0, self.plate_height - 1]], dtype=np.float32)
        
        # 添加 filtered_detections 屬性，初始化為 None
        self.filtered_detections = None
    
    def detect(self, img):

        boxes, confs, clses, keypoints = None, None, None, None
        # Run YOLOv8 on the frame
        results = self.model.predict(img, agnostic_nms=True, device=self.device)
        if results and results[0].boxes:
            # Get the boxes and track IDs
            boxes = results[0].boxes.xyxy.cpu()
            confs = results[0].boxes.conf.cpu().reshape(-1, 1)
            clses = results[0].boxes.cls.cpu().reshape(-1, 1)
            if self.is_pose_model:
                keypoints = results[0].keypoints.xy.cpu().reshape(-1, 8)

        # Visualize the results on the frame
        # annotated_frame = results[0].plot()
        
        if boxes is not None and confs is not None and clses is not None:
            if self.is_pose_model:
                detections = torch.cat((boxes, confs, clses, keypoints), 1)
            else:
                detections = torch.cat((boxes, confs, clses), 1)
            filtered_detections = detections[detections[:, 4] > self.confidence_threshold]
            self.filtered_detections = filtered_detections 
            return filtered_detections
        
        return torch.zeros(14).view(1, 14) if self.is_pose_model else torch.zeros(6).view(1, 6)
    
    def tilt_correct(self, img, corners):
        corners = np.array(corners, dtype=np.float32).reshape(-1, 2)
        #print(corners)
        
        # 計算透視變換矩陣
        transform_matrix = cv2.getPerspectiveTransform(corners, self.destination_corners)

        # 執行透視校正
        corrected_image = cv2.warpPerspective(img, transform_matrix, (self.plate_width, self.plate_height))
        # cv2.imshow("corrected_image", corrected_image)
        # cv2.waitKey(0)

        return corrected_image


class charInference(yolov8Inference):
    plate_regular_dict = {
        'D1A2D3': 3, 'D2A2': 2, 'D3A1D1': 3, 'D3A2': 3, 'D3A3': 3, 'D4A1D1': 4, 'D4A2': 4, 'A2D2': 2, 'A2D3': 2, 'A2D4': 2, 'A3D3': 3, 'A3D4': 3, 
        'A1D5': 2, 'A1D4': 2, 'D1A1D4': 2, 'D6': 4, 'D2A1D1': 2, 'A1D1A1D3': 3
    }
    def __init__(self, weights_path, device, class_dict, confidence_threshold):
        # load YOLOv8 model
        self.model = YOLO(weights_path, task="detect") #.to(device=device)
        self.device = device
        self.class_dict = class_dict
        self.confidence_threshold = confidence_threshold
        # 定義車牌的寬和高
        self.plate_width, self.plate_height = 416, 208

        # 定義校正後的四個角點坐標
        self.destination_corners = np.array([[0, 0], [self.plate_width - 1, 0], [self.plate_width - 1, self.plate_height - 1], [0, self.plate_height - 1]], dtype=np.float32)
        
        # 添加 filtered_detections 屬性，初始化為 None
        self.filtered_detections = None
    
    def detect(self, online_plate, online_targets, seen_plates, temp_plate_info, is_stream, fps):
        online_plate_num = []
        # temp_plate_info = []
        # temp_plate_info = {}
        # Run YOLOv8 on the plate set
        results = self.model.predict(online_plate, agnostic_nms=True, device=self.device, imgsz=416)
        for char_result, t, plate_image in zip(results, online_targets, online_plate):
            id = t.track_display_id
            char_x_dist = {}

            if char_result and char_result.boxes:
                boxes = char_result.boxes.xywh.cpu().tolist()  # Boxes object for bbox outputs
                classes = char_result.boxes.cls.cpu().tolist()    # Boxes object for class outputs

                for box, cls in zip(boxes, classes):
                    x, y, w, h = box
                    char_x_dist[x] = self.class_dict[int(cls)]
                
                plate_number = ''
                for c in dict(sorted(char_x_dist.items())).values():
                    plate_number += c
                
                # 去除 符號 - 與不可能的車牌
                plate_number = self.correct_plate_number(plate_number)
                
                if len(plate_number) > 5 and (not t.is_confirm or t.direction != "forward"):
                    t.set_track_plate_number(plate_number)
                try:
                    # 取得該車牌id出現最多次的車牌號碼
                    color, plate_number = t.track_plate_number
                    if t.is_confirm and plate_number not in seen_plates:
                        # t.set_track_plate_number(plate_number)
                        # temp_plate_info.append((id, plate_number))

                        # 如果 temp_plate_info 中不存在這個 id，或者 id 對應的車牌號碼不同於新的車牌號碼則更新
                        if not temp_plate_info.get(id) or temp_plate_info[id][0] != plate_number:
                            if temp_plate_info.get(id):
                                # 從 seen_plates 中移除原來的車牌號碼
                                seen_plates.remove(temp_plate_info[id][0])
                                start_time = temp_plate_info[id][2]
                            else:
                                if is_stream:
                                    start_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                                else:
                                    start_time = self.frame_to_time_string(t.get_start_frame, fps)
                            
                            # 更新 temp_plate_info 中的車牌資訊
                            temp_plate_info[id] = (plate_number, array_to_base64(plate_image), start_time)
                            # 將新的車牌號碼加入 seen_plates
                            seen_plates.add(plate_number)

                except:
                    plate_number = ''
                    color = (0, 255, 0)
                online_plate_num.append((color, plate_number))
            else:
                try:
                    color, plate_number = t.track_plate_number
                except:
                    plate_number = ''
                    color = (0, 255, 0)
                online_plate_num.append((color, plate_number))   

        # if temp_plate_info != []:
        #     print(temp_plate_info)
        #     logging.debug(f'車牌號碼：{temp_plate_info}')
        # print(temp_plate_info)
        print(seen_plates)
        return online_plate_num, temp_plate_info
    
    def correct_plate_number(self, plate_number):
        plate_number = re.sub(r'[-]+', '', plate_number)
        regular_plate_number = self.regular_expression_plate_number(plate_number)

        if regular_plate_number in self.plate_regular_dict:
            position_to_insert = self.plate_regular_dict[regular_plate_number]
            plate_number = plate_number[:position_to_insert] + '-' + plate_number[position_to_insert:]
        else:
            return ''
        
        return plate_number
    
    def regular_expression_plate_number(self, plate_number):
        regular_plate_number = ''
        temp = 0
        for n in plate_number:
            if n.isalpha():
                marker = 'A'
            elif n.isdigit():
                marker = 'D'
            else:
                continue

            if not regular_plate_number or regular_plate_number[-1] != marker:
                if temp != 0:
                    regular_plate_number += str(temp)
                    temp = 0
                regular_plate_number += marker
            temp += 1
            
        regular_plate_number += str(temp)
        return regular_plate_number
    
    def frame_to_time_string(self, frame_number, fps):
        total_seconds = frame_number / fps
        hours = int(total_seconds / 3600)
        minutes = int((total_seconds % 3600) / 60)
        seconds = int(total_seconds % 60)
        return '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)

class LPRSystem:
    class_dict = {0: 'plate'}
    char_dict = {
                    0: '-',
                    1: '0',  2: '1',  3: '2',  4: '3',  5: '4',  6: '5',  7: '6',  8: '7',  9: '8',
                    10: '9', 11: 'A', 12: 'B', 13: 'C', 14: 'D', 15: 'E', 16: 'F', 17: 'G', 18: 'H',
                    19: 'I', 20: 'J', 21: 'K', 22: 'L', 23: 'M', 24: 'N', 25: 'O', 26: 'P', 27: 'Q',
                    28: 'R', 29: 'S', 30: 'T', 31: 'U', 32: 'V', 33: 'W', 34: 'X', 35: 'Y', 36: 'Z'
                }
    
    def __init__(self, args, device, plate_confidence_threshold=0.5, char_confidence_threshold=0.5):
        self._args = args
        self.plate_model = plateInference(self._args.plate_weight_path, device, self.class_dict, plate_confidence_threshold)
        self.char_model = charInference(self._args.char_weight_path, device, self.char_dict, char_confidence_threshold)
    
    def detect_plate_in_video(self, conn, cursor, save=False, socketio=None, save_org=False):
        seen_plates = set()
        temp_plate_info = {}
        plate_info = None
        is_stream =  False
        frame_id = 0
        det_tilt = []
        location = get_location()
        plate_dict = {}
        plate_time_dict = {}
        time_interval = timedelta(minutes=1) # 如果出現後消失Ｎ分鐘後又出現則紀錄
        max_dict_size = 1000  # 設定字典的最大大小

        # Open the video file

        if self._args.video_path.startswith('rtsp://'):
            is_stream = True
            cap = VideoCapture(self._args.video_path)
            current_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            # output = "app/static/uploads/" + current_time + "_detect.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            print(self._args.video_path)
            cap = cv2.VideoCapture(self._args.video_path, cv2.CAP_GSTREAMER)
            # cap = cv2.VideoCapture(self._args.video_path)
            # output = self._args.video_path[:-4] + "_detect.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        output = self._args.output_path
        tracker = BYTETracker(self._args, frame_rate=self._args.fps)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS = int(cap.get(cv2.CAP_PROP_FPS))
        print(f'FPS : {FPS}')
        current_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        if save:
            out = av.open(output, 'w')
            stream = out.add_stream('libx264', str(FPS))
            stream.bit_rate = 8000000   

        if save_org:
            original_output_path = f"{current_time}_original_video.mp4"
            original_out = cv2.VideoWriter(original_output_path, fourcc, FPS, (width, height))

        start_time = time.time()

        # Loop through the video frames
        while cap.isOpened():
            #print("open success")
            # Read a frame from the vide
            success, frame = cap.read()
            if save_org:
                original_out.write(frame)

            if success:
                tilt_det = self.plate_model.detect(frame)
                # det = tilt_det[:, :6]
                # print(det.shape)
                # filtered_detections = self.plate_model.filtered_detections
                # if filtered_detections is not None:
                #     det_tilt = self.plate_model.tilt_correct(frame, filtered_detections)

                #det=self.plate_model.detect(frame)
                #online_targets = tracker.update(det, (height, width), (height, width))

                tilt_targets = tracker.update(tilt_det, (height, width), (height, width))
                
                online_tlwhs = []
                online_ids = []
                online_scores = []
                online_plate = []
                online_tkp = []
                online_plate_num = []
                plate_tilt = []

                for t in tilt_targets:
                    tlwh = t.detect_tlwh
                    # vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > self._args.min_box_area: # and not vertical:
                        if t.is_display == False:
                            t.display()
                        tid = t.track_display_id
                        xmin, ymin, w, h = int(tlwh[0]), int(tlwh[1]), int(tlwh[2]), int(tlwh[3])
                        plate_img = frame[ymin:ymin + h, xmin:xmin + w]
                        if t.keypoints is not None:
                            tkp = t.keypoints
                            tilt_img = self.plate_model.tilt_correct(frame, tkp)
                            online_plate.append(tilt_img)
                            online_tkp.append(tkp)
                        else:
                            if min(plate_img.shape) <= 0:
                                continue
                        
                            online_plate.append(plate_img)
                        
                        # cv2.imshow("before_corrected_image", cv2.resize(plate_img, (416, 208)))
                        # cv2.waitKey(0)
                        # print(plate_img)

                        # plate_tilt.append(tilt_targets)
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)

                # Run char model
                if len(online_plate) > 0:
                    #online_plate_num = self.char_model.detect(online_plate, online_targets, seen_plates)
                    tilt_plate_num, plate_info = self.char_model.detect(online_plate, tilt_targets, seen_plates, temp_plate_info, is_stream, FPS)
                else:
                    tilt_plate_num = []

                elapsed_time = time.time() - start_time
                fps = 1 / elapsed_time
                start_time = time.time()

                #online_img = self.plot_tracking(frame, online_tlwhs, online_ids, frame_id=frame_id, fps=fps, plate=online_plate_num)
                online_img = self.plot_tracking(frame, online_tlwhs, online_ids, tkps=online_tkp, frame_id=frame_id, fps=fps, plate=tilt_plate_num)
                frame_id += 1

                # 調整大小、置中
                new_width = 1920
                new_height = 1080
                resized_img = cv2.resize(online_img, (new_width, new_height))
                cv2.namedWindow("YOLOv8 Tracking", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("YOLOv8 Tracking", new_width, new_height)
                screen_width = 1920
                screen_height = 1080 
                x_pos = int((screen_width - new_width) / 2)
                y_pos = int((screen_height - new_height) / 2)
                cv2.moveWindow("YOLOv8 Tracking", x_pos, y_pos)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(resized_img, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (135, 130, 226), 2, cv2.LINE_AA)
                cv2.imshow("YOLOv8 Tracking", resized_img)
                # cv2.waitKey(0)

                # save frame
                if save:
                    # out.write(online_img)
                    frame = av.VideoFrame.from_ndarray(online_img, format='bgr24')
                    packet = stream.encode(frame)
                    out.mux(packet)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if socketio is not None and frame_id % 10 == 0:
                    online_img = resize_to_max_edge(online_img)
                    if self._args.video_path.startswith('rtsp://'):
                        socketio.emit('streamInferenceResult', {'plate_info': plate_info}, namespace='/streaming')
                        socketio.emit('streamImageResult', {'streaming_image': array_to_base64(online_img)}, namespace='/streaming')
                    else:
                        socketio.emit('inferenceResult', {'plate_info': plate_info}, namespace='/video/demo')
                        socketio.emit('imageResult', {'image': array_to_base64(online_img)}, namespace='/video/demo')
                if plate_info is not None:
                    plates = [info[0] for info in plate_info.values()]
                    for plate in plates:
                        current_time = datetime.now()
                        if plate in plate_dict.keys():
                            last_seen_time = plate_time_dict[plate]
                            if current_time - last_seen_time > time_interval:
                                #logging.info(f"Plate reappeared after interval: {plate}")
                                plate_time_dict[plate] = current_time
                                update_plate_last_seen_time(cursor, plate, location, current_time)
                                conn.commit()
                            else:
                                logging.debug(f"Plate already seen, not updating last seen time: {plate}")
                            plate_dict[plate] += 1
                        else:
                            if len(plate_dict) >= max_dict_size:
                                oldest_plate = next(iter(plate_dict))
                                del plate_dict[oldest_plate]
                                del plate_time_dict[oldest_plate]

                            plate_dict[plate] = 1
                            logging.info(f"New plate: {plate}")
                            plate_time_dict[plate] = current_time
                            insert_plate_num(cursor, plate, location)
                            conn.commit()

            else:
                # Break the loop if the end of the video is reached
                break

        BaseTrack.reset_id()
        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()

        # flush
        packet = stream.encode(None)
        out.mux(packet)

        out.close()

        if save_org:
            original_out.release()


        return plate_info

    # Create a new image to ensure it does not affect the original image
    def plot_tracking(self, image, tlwhs, obj_ids, tkps=[], scores=None, frame_id=0, fps=0., ids2=None, plate=[]):
        im = np.ascontiguousarray(np.copy(image))
        im_h, im_w = im.shape[:2]

        # Define parameters for font scale, text scaling, thickness, and line thickness
        font_scale = im.shape[1] / 500
        text_scale = font_scale
        text_thickness = 3
        line_thickness = 1

        # Calculate the radius for rendering circles
        radius = max(5, int(im_w/140.))
        '''
        # Add frame number, FPS, and the number of tracked targets in the top-left corner
        cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                    (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, font_scale, (255, 0, 255), thickness=3)
                    '''
        # Iterate over the coordinates list of tracked targets
        for i, tlwh in enumerate(tlwhs):
            x1, y1, w, h = tlwh
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))  # Calculate integer coordinates of the target
            obj_id = int(obj_ids[i])  # Get the ID of the target
            id_text = '{}'.format(int(obj_id))  # Convert ID to text
            if ids2 is not None:
                id_text = id_text + ', {}'.format(int(ids2[i]))  # If there is a second ID, add the second ID

            # If there is license plate information, add the license plate text
            if len(plate) > 0:
                id_text = id_text + " " + plate[i][1]

            # Calculate the width and height of the text as well as the baseline
            (text_width, text_height), baseline = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_PLAIN, text_scale, text_thickness)

            # Calculate the starting point of the text, ensuring that the text does not go beyond the top boundary
            start_x = max(intbox[0], 0)
            start_y = max(intbox[1], text_height)

            # Ensure that the text does not go beyond the right boundary
            if start_x + text_width > im_w:
                start_x = im_w - text_width

            # Set the starting point of the text
            start_point = (start_x, start_y)

            # Draw the bounding box of the target
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=(255, 0, 0), thickness=line_thickness)

            if len(tkps) >= i + 1:
                tkp = np.array(tkps[i], dtype=np.int32).reshape(-1, 2)
                # Draw the four sides of the target
                cv2.polylines(im, [tkp], isClosed=True, color=(0, 0, 255), thickness=line_thickness)
            
            # Add text to the image
            cv2.putText(im, id_text, start_point, cv2.FONT_HERSHEY_PLAIN, text_scale, plate[i][0],
                        thickness=text_thickness)
                        

        return im
    
    @staticmethod
    def make_parser():
        parser = argparse.ArgumentParser("ByteTrack Demo!")
        # parser.add_argument(
        #     "demo", default="image", help="demo type, eg. image, video and webcam"
        # )
        parser.add_argument("-expn", "--experiment-name", type=str, default=None)
        parser.add_argument("-n", "--name", type=str, default=None, help="model name")

        parser.add_argument(
            #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
            "--path", default="./videos/palace.mp4", help="path to images or video"
        )

        # exp file
        parser.add_argument(
            "-f",
            "--exp_file",
            default=None,
            type=str,
            help="pls input your expriment description file",
        )
        parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
        parser.add_argument(
            "--device",
            default="gpu",
            type=str,
            help="device to run our model, can either be cpu or gpu",
        )
        parser.add_argument("--conf", default=None, type=float, help="test conf")
        parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
        parser.add_argument("--tsize", default=None, type=int, help="test img size")
        parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
        parser.add_argument(
            "--fp16",
            dest="fp16",
            default=False,
            action="store_true",
            help="Adopting mix precision evaluating.",
        )
        parser.add_argument(
            "--fuse",
            dest="fuse",
            default=False,
            action="store_true",
            help="Fuse conv and bn for testing.",
        )
        parser.add_argument(
            "--trt",
            dest="trt",
            default=False,
            action="store_true",
            help="Using TensorRT model for testing.",
        )
        # tracking args
        parser.add_argument("--track_thresh", type=float, default=0.3, help="tracking confidence threshold")  
        parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
        parser.add_argument("--match_thresh", type=float, default=0.95, help="matching threshold for tracking")
        parser.add_argument("--second_match_thresh", type=float, default=0.95, help="second matching threshold for tracking")
        parser.add_argument("--new_match_thresh", type=float, default=0.95, help="new matching threshold for tracking")
        parser.add_argument("--iou_type", type=str, default='diou', help="iou type for match")
        parser.add_argument(
            "--aspect_ratio_thresh", type=float, default=1.6,
            help="threshold for filtering out boxes of which aspect ratio are above the given value."
        )
        parser.add_argument('--min_box_area', type=float, default=0, help='filter out tiny boxes')
        parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

        parser.add_argument('--plate_weight_path', type=str, default='./static/weights/yolov8n_pose.engine', help='plate weight path')
        parser.add_argument("--char_weight_path", type=str, default='./static/weights/yolov8n_char_plus.engine', help="char weight path")

        parser.add_argument('--video_path', type=str, default="nvarguscamerasrc ! nvvidconv ! video/x-raw, width=1920, height=1080, framerate=30/1, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink", help='video path')
        parser.add_argument('--output_path', type=str, default="detect.mp4", help='output path')

        return parser
    
    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, new_args):
        self._args = new_args


if __name__ == '__main__':
    # args = LPRSystem.make_parser().parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = LPRSystem.make_parser().parse_args()

    # 初始化 LPR 系統
    lpr_system = LPRSystem(args, device, plate_confidence_threshold=0.5, char_confidence_threshold=0.5)

    # 在視頻中檢測車牌並保存
    plate_info = lpr_system.detect_plate_in_video(save=False)

    
    
    
    
    
