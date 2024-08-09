#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys
import os
import platform
import configparser
import math
import argparse
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GLib, Gst, GstRtspServer, GObject
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import PERF_DATA
import time
import pyds
import re
from collections import defaultdict
import os.path
from os import path
import numpy as np
import cv2
from update_epc_mysql import *
from log_config import setup_logging, get_location
import logging

setup_logging(enable_logging=True)
conn, cursor = setup_database()
plate_dict = {}
plate_time_dict = {}

track_plate_count = defaultdict(lambda: [defaultdict(lambda: 0), False])
no_display = False
silent = False
file_loop = False
perf_data = None

TILED_OUTPUT_WIDTH=1280
TILED_OUTPUT_HEIGHT=720
PRIMARY_DETECTOR_UID = 1
SECONDARY_DETECTOR_UID = 2
PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3
past_tracking_meta=[0]
last = time.time()

plate_regular_dict = {
    'D1A2D3': 3, 'D2A2': 2, 'D3A1D1': 3, 'D3A2': 3, 'D3A3': 3, 'D4A1D1': 4, 'D4A2': 4, 'A2D2': 2, 'A2D3': 2, 'A2D4': 2, 'A3D3': 3, 'A3D4': 3, 
     'A1D5': 2, 'A1D4': 2, 'D1A1D4': 2, 'D6': 4, 'D2A1D1': 2, 'A1D1A1D3': 3
}

# Function to create an RTSP server for a specific stream
def create_rtsp_server(rtsp_port_num, updsink_port_num, codec, stream_id):
    server = GstRtspServer.RTSPServer.new()
    server.props.service = f"{rtsp_port_num}"
    server.attach(None)

    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch(f"( udpsrc name=pay0 port={updsink_port_num} buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string){codec}, payload=96 \" )")
    factory.set_shared(True)
    mount_point = f"/ds-test-{stream_id}"
    server.get_mount_points().add_factory(mount_point, factory)
    
    print(f"\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:{rtsp_port_num}{mount_point} ***\n\n")
    
def update_track_plate_count(object_id, new_plate_number): 
    global track_plate_count
    track_plate_dict = track_plate_count[object_id][0]
    track_plate_dict[new_plate_number] += 1
    if track_plate_dict[new_plate_number] > 5:
        track_plate_count[object_id][1] = True
    
    max_track_plate_number = max(track_plate_dict, key=lambda k: track_plate_dict[k])
    
    return max_track_plate_number, track_plate_count[object_id][1]

            
def regular_expression_plate_number(plate_number):
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
        
def correct_plate_number(plate_number):
    plate_number = re.sub(r'[-]+', '', plate_number)
    regular_plate_number = regular_expression_plate_number(plate_number)

    if regular_plate_number in plate_regular_dict:
        position_to_insert = plate_regular_dict[regular_plate_number]
        plate_number = plate_number[:position_to_insert] + plate_number[position_to_insert:]
    else:
        return ''
        
    return plate_number

def crop_object(image, obj_meta):
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)

    crop_img = image[top:top+height, left:left+width]
	
    return crop_img
    
def streamdemux_sink_pad_buffer_probe(pad, info, u_data):
    global last
    frame_number=0
    #Intiallizing object counter with 0.
    obj_counter = {
        PGIE_CLASS_ID_VEHICLE:0,
        PGIE_CLASS_ID_PERSON:0,
        PGIE_CLASS_ID_BICYCLE:0,
        PGIE_CLASS_ID_ROADSIGN:0
    }
    char_dict = {
                    0: '-',
                    1: '0',  2: '1',  3: '2',  4: '3',  5: '4',  6: '5',  7: '6',  8: '7',  9: '8',
                    10: '9', 11: 'A', 12: 'B', 13: 'C', 14: 'D', 15: 'E', 16: 'F', 17: 'G', 18: 'H',
                    19: 'I', 20: 'J', 21: 'K', 22: 'L', 23: 'M', 24: 'N', 25: 'O', 26: 'P', 27: 'Q',
                    28: 'R', 29: 'S', 30: 'T', 31: 'U', 32: 'V', 33: 'W', 34: 'X', 35: 'Y', 36: 'Z'
                }
    num_rects=0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        plate_dict = {}
        now = time.time()
        text = ""
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            if obj_meta.unique_component_id == PRIMARY_DETECTOR_UID:
                # print("Plate: ", obj_meta.class_id, obj_meta.confidence)
                obj_counter[obj_meta.class_id] += 1
                try: 
                    l_obj=l_obj.next
                except StopIteration:
                    break
            elif obj_meta.unique_component_id == SECONDARY_DETECTOR_UID:
                if obj_meta.parent:
                    if obj_meta.parent.object_id not in plate_dict:
                        plate_dict[obj_meta.parent.object_id] = {}
                        
                    obj_meta.rect_params.border_width = 0
                    obj_meta.text_params.display_text = ""
                    
                    plate_dict[obj_meta.parent.object_id][obj_meta.rect_params.left] = char_dict[obj_meta.class_id]
                    # print("Char found for parent object", obj_meta.parent.object_id, char_dict[obj_meta.class_id], obj_meta.confidence)
                    p_left, p_top, p_width, p_height = obj_meta.parent.rect_params.left, obj_meta.parent.rect_params.top, obj_meta.parent.rect_params.width, obj_meta.parent.rect_params.height
                    left, top, width, height = obj_meta.rect_params.left, obj_meta.rect_params.top, obj_meta.rect_params.width, obj_meta.rect_params.height
                    # print("Parents Left: {} Top: {} Width: {} Height: {}".format(obj_meta.parent.rect_params.left, obj_meta.parent.rect_params.top, obj_meta.parent.rect_params.width, obj_meta.parent.rect_params.height))
                    # print("Left: {} Top: {} Width: {} Height: {}".format(obj_meta.rect_params.left, obj_meta.rect_params.top, obj_meta.rect_params.width, obj_meta.rect_params.height))
                    # if left < p_left or top < p_top or left + width > p_left + p_width or top + height > p_top + p_height:
                    #    print("false")
                else:
                    print("No object parent")
                text += char_dict[obj_meta.class_id]
                l_obj=l_obj.next
                
        for object_id, value in plate_dict.items():
            plate_number = ''
            for c in dict(sorted(plate_dict[object_id].items())).values():
                plate_number += c
            plate_number = correct_plate_number(plate_number)
            plate_number, confirmed = update_track_plate_count(object_id, plate_number)
            plate_dict[object_id] = [plate_number, confirmed]
            
        l_obj=frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            if obj_meta.unique_component_id == PRIMARY_DETECTOR_UID:
                try:
                    if obj_meta.object_id in plate_dict and plate_dict[obj_meta.object_id][0] != '':
                        display_text = str(obj_meta.object_id) + " " + plate_dict[obj_meta.object_id][0]
                        obj_meta.text_params.display_text = display_text
                        #obj_meta.text_params.x_offset = 10
                        #print(pyds.get_string(obj_meta.text_params.y_offset))
                        obj_meta.text_params.y_offset = max(0, obj_meta.text_params.y_offset - 50)
                        obj_meta.text_params.font_params.font_size = 20
                        if plate_dict[obj_meta.object_id][1]:
                            obj_meta.text_params.font_params.font_color.set(1.0, 0.0, 0.0, 1.0)
                        else:
                            obj_meta.text_params.font_params.font_color.set(0.0, 1.0, 0.0, 1.0)
                        obj_meta.text_params.set_bg_clr = 0
                        
                        #if not plate_dict[obj_meta.object_id][1]:
                        #    obj_meta.text_params.display_text = ""

                    else:
                        obj_meta.rect_params.border_width = 0
                        obj_meta.text_params.display_text = ""
                        #display_text = str(obj_meta.object_id) + " " + plate_dict[obj_meta.object_id][0]
                        #obj_meta.text_params.display_text = display_text
                    l_obj=l_obj.next
                except StopIteration:
                    break
            elif obj_meta.unique_component_id == SECONDARY_DETECTOR_UID:
                l_obj=l_obj.next
        # print(text)
        # print(plate_dict)
        
        # update frame rate through this probe
        stream_index = "stream{0}".format(frame_meta.pad_index)
        global perf_data
        perf_data.update_fps(stream_index)
        
        frame_rate = perf_data.get_fps(stream_index)  #round(1 / (now - last), 2)
        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        py_nvosd_text_params.display_text = "Frame Number={} FPS={} Number of Objects={} Plate_count={}".format(frame_number, frame_rate, num_rects, obj_counter[PGIE_CLASS_ID_VEHICLE])

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        # print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
            
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
        last = now
    return Gst.PadProbeReturn.OK
    
def filter1_src_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    global conn
    global cursor
    global plate_dict
    global plate_time_dict
    time_interval = timedelta(minutes=1)
    max_dict_size = 1000
    location = get_location()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list
        
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            if obj_meta.unique_component_id == PRIMARY_DETECTOR_UID:
                try:
                    if obj_meta.text_params.display_text != '':
                        # 在這裡進行對辨識出的車牌號碼進行處理
                        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                        n_frame = crop_object(n_frame, obj_meta)
                        # convert python array into numpy array format in the copy mode.
                        frame_copy = np.array(n_frame, copy=True, order='C')
                        # convert the array into cv2 default color format
                        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGRA)
                        if is_aarch64(): # If Jetson, since the buffer is mapped to CPU for retrieval, it must also be unmapped 
                            pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id) # The unmap call should be made after operations with the original array are complete.
                                                                                            #  The original array cannot be accessed after this call.
                        display_text = pyds.get_string(obj_meta.text_params.display_text)
                        img_path = "{}/stream_{}/{}.jpg".format(folder_name, frame_meta.pad_index, display_text)
                        # display_text 為辨識車牌結果 "id plate_number", frame_copy 為 車牌圖片
                        # cv2.imwrite(img_path, frame_copy)

                        if display_text is not None:
                            parts = display_text.split()
                            if len(parts) > 1:
                                plate = parts[-1]
                                current_time = datetime.now()
                                if plate in plate_dict.keys():
                                    last_seen_time = plate_time_dict[plate]
                                    if current_time - last_seen_time > time_interval:
                                        #logging.info(f"Plate reappeared after interval: {plate}")
                                        plate_time_dict[plate] = current_time
                                        print("plate_time_dict : ", plate_time_dict)
                                        update_plate_last_seen_time(cursor, plate, location, current_time)
                                        conn.commit()
                                    else:
                                        logging.debug(f"Plate already seen, not updating last seen time: {plate}")
                                    plate_dict[plate] += 1
                                else:
                                    if len(plate_dict) >= max_dict_size:
                                        print("len(plate_dict) : ", len(plate_dict))
                                        oldest_plate = next(iter(plate_dict))
                                        del plate_dict[oldest_plate]
                                        del plate_time_dict[oldest_plate]

                                    plate_dict[plate] = 1
                                    logging.info(f"New plate: {plate}")
                                    plate_time_dict[plate] = current_time
                                    insert_plate_num(cursor, plate, location)
                                    conn.commit()
                    l_obj=l_obj.next
                except StopIteration:
                    break
            elif obj_meta.unique_component_id == SECONDARY_DETECTOR_UID:
                l_obj=l_obj.next
            
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK	

def cb_newpad(decodebin, decoder_src_pad,data):
    print("In cb_newpad\n")
    caps=decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()
    source_bin=data
    features=caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=",gstname)
    if(gstname.find("video")!=-1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=",features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy,Object,name,user_data):
    print("Decodebin child added:", name, "\n")
    if(name.find("decodebin") != -1):
        Object.connect("child-added",decodebin_child_added,user_data)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property('drop-on-latency') != None:
            Object.set_property("drop-on-latency", True)



def create_csi_source_bin(index, dev):
    print("Creating csi source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")
        return None

    # Source element for reading from the CSI camera
    source = Gst.ElementFactory.make("nvarguscamerasrc", "csi-camera-source")
    # source.set_property('bufapi-version', True)
    sensor_id = int(dev)
    source.set_property('sensor-id', sensor_id)
    source.set_property('sensor-mode', 1) #1:3840*2160 2:1920*1080
    
    if not source:
        sys.stderr.write(" Unable to create source element \n")
    
    crop = Gst.ElementFactory.make("videocrop", "crop")
    crop.set_property("top", 180)
    crop.set_property("bottom", 180)
    crop.set_property("left", 320)
    crop.set_property("right", 320)
   
    # nvvideoconvert to convert incoming raw buffers to NVMM Mem (NvBufSurface API)
    nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_src")
    if not nvvidconvsrc:
        sys.stderr.write(" Unable to create Nvvideoconvert \n")
        return None  
        
    nvvidconvsrc2 = Gst.ElementFactory.make("nvvideoconvert", "convertor_crop")
    if not nvvidconvsrc2:
        sys.stderr.write(" Unable to create Nvvideoconvert \n")
        return None  

    queue_name = "srcqueue_%02d" % index
    queue = Gst.ElementFactory.make("queue", queue_name)

    caps_vidconvsrc = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
    
    # Set the desired output resolution
    output_width = 640
    output_height = 360
    framerate = "30/1"
    format = "NV12"
    
    # Adjust these parameters to implement digital zoom
    zoom_factor = 1  # Adjust as needed for zoom level
    input_width = output_width * zoom_factor
    input_height = output_height * zoom_factor
    
    caps_vidconvsrc.set_property('caps', Gst.Caps.from_string(
        f"video/x-raw(memory:NVMM), width={input_width}, height={input_height}, framerate={framerate}, format={format}"
    ))
    if not caps_vidconvsrc:
        sys.stderr.write(" Unable to create capsfilter \n")
        return None
    
    caps_vidconvsrc.connect("pad-added", cb_newpad, nbin)
    # source.connect("child-added", decodebin_child_added, nbin)
    
    # Add elements to bin
    Gst.Bin.add(nbin, source)
    Gst.Bin.add(nbin, crop)
    Gst.Bin.add(nbin, nvvidconvsrc)
    Gst.Bin.add(nbin, nvvidconvsrc2)
    Gst.Bin.add(nbin, queue)
    Gst.Bin.add(nbin, caps_vidconvsrc)

    # Link the elements together
    if not source.link(nvvidconvsrc):
        sys.stderr.write("Failed to link source to nvvidconvsrc\n")
        return None
    if not nvvidconvsrc.link(crop):
        sys.stderr.write("Failed to link nvvidconvsrc to crop\n")
        return None
    if not crop.link(nvvidconvsrc2):
        sys.stderr.write("Failed to link crop to nvvidconvsrc2\n")
        return None
    if not nvvidconvsrc2.link(queue):
        sys.stderr.write("Failed to link nvvidconvsrc2 to queue\n")
        return None
    if not queue.link(caps_vidconvsrc):
        sys.stderr.write("Failed to link queue to caps_vidconvsrc\n")
        return None
        
    pad = caps_vidconvsrc.get_static_pad("src")
    ghostpad = Gst.GhostPad.new("src",pad)
    bin_pad=nbin.add_pad(ghostpad)
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    
    return nbin
    
def create_usb_source_bin(index, dev):
    print("Creating usb source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")
        return None

    # Source element for reading from the USB camera
    source = Gst.ElementFactory.make("v4l2src", "usb-camera-source")
    source.set_property('device', dev)
    
    if not source:
        sys.stderr.write(" Unable to create source element \n")

    # Convert video format
    capsfilter = Gst.ElementFactory.make("capsfilter", "v4l2src_caps")
    caps = Gst.Caps.from_string("video/x-raw, framerate=30/1")
    capsfilter.set_property("caps", caps)
    
    # videoconvert to make sure a superset of raw formats are supported
    vidconvsrc = Gst.ElementFactory.make("videoconvert", "convertor_src1")
    
    if not vidconvsrc:
        sys.stderr.write(" Unable to create videoconvert \n")
        return None

    # nvvideoconvert to convert incoming raw buffers to NVMM Mem (NvBufSurface API)
    nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_src2")
    if not nvvidconvsrc:
        sys.stderr.write(" Unable to create Nvvideoconvert \n")
        return None

    caps_vidconvsrc = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
    caps_vidconvsrc.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1, format=NV12"))
    if not caps_vidconvsrc:
        sys.stderr.write(" Unable to create capsfilter \n")
        return None
    
    caps_vidconvsrc.connect("pad-added", cb_newpad, nbin)
    # source.connect("child-added", decodebin_child_added, nbin)
    
    # Add elements to bin
    Gst.Bin.add(nbin, source)
    Gst.Bin.add(nbin, capsfilter)
    Gst.Bin.add(nbin, vidconvsrc)
    Gst.Bin.add(nbin, nvvidconvsrc)
    Gst.Bin.add(nbin, caps_vidconvsrc)

    if not source.link(capsfilter):
        sys.stderr.write("Failed to link source to capsfilter\n")
        return None
    if not capsfilter.link(vidconvsrc):
        sys.stderr.write("Failed to link capsfilter to vidconvsrc\n")
        return None
    if not vidconvsrc.link(nvvidconvsrc):
        sys.stderr.write("Failed to link vidconvsrc to nvvidconvsrc\n")
        return None
    if not nvvidconvsrc.link(caps_vidconvsrc):
        sys.stderr.write("Failed to link nvvidconvsrc to caps_vidconvsrc\n")
        return None
        
    pad = caps_vidconvsrc.get_static_pad("src")
    ghostpad = Gst.GhostPad.new("src", pad)
    bin_pad=nbin.add_pad(ghostpad)
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    
    return nbin

def create_ids_source_bin(index, dev):
    print("Creating ids source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")
        return None

    # Source element for reading from the USB camer
    source = Gst.ElementFactory.make("aravissrc", "ids-camera-source")
    source.set_property('camera-name', dev)
    
    if not source:
        sys.stderr.write(" Unable to create source element \n")
    
    bayer2rgb = Gst.ElementFactory.make("bayer2rgb", "bayer2rgb")
    
    # Convert video format
    capsfilter = Gst.ElementFactory.make("capsfilter", "idssrc_caps")
    caps = Gst.Caps.from_string("video/x-raw, width=640, height=480, framerate=30/1")
    capsfilter.set_property("caps", caps)
    
    # videoconvert to make sure a superset of raw formats are supported
    vidconvsrc = Gst.ElementFactory.make("videoconvert", "convertor_src1")
    
    if not vidconvsrc:
        sys.stderr.write(" Unable to create videoconvert \n")
        return None

    # nvvideoconvert to convert incoming raw buffers to NVMM Mem (NvBufSurface API)
    nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_src2")
    if not nvvidconvsrc:
        sys.stderr.write(" Unable to create Nvvideoconvert \n")
        return None

    caps_vidconvsrc = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
    caps_vidconvsrc.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1, format=NV12"))
    if not caps_vidconvsrc:
        sys.stderr.write(" Unable to create capsfilter \n")
        return None
    
    caps_vidconvsrc.connect("pad-added", cb_newpad, nbin)
    # source.connect("child-added", decodebin_child_added, nbin)
    
    # Add elements to bin
    Gst.Bin.add(nbin, source)
    Gst.Bin.add(nbin, bayer2rgb)
    Gst.Bin.add(nbin, capsfilter)
    Gst.Bin.add(nbin, vidconvsrc)
    Gst.Bin.add(nbin, nvvidconvsrc)
    Gst.Bin.add(nbin, caps_vidconvsrc)

    if not source.link(bayer2rgb):
        sys.stderr.write("Failed to link source to bayer2rgb\n")
        return None
    if not bayer2rgb.link(capsfilter):
        sys.stderr.write("Failed to link bayer2rgb to capsfilter\n")
        return None
    if not capsfilter.link(vidconvsrc):
        sys.stderr.write("Failed to link capsfilter to vidconvsrc\n")
        return None
    if not vidconvsrc.link(nvvidconvsrc):
        sys.stderr.write("Failed to link vidconvsrc to nvvidconvsrc\n")
        return None
    if not nvvidconvsrc.link(caps_vidconvsrc):
        sys.stderr.write("Failed to link nvvidconvsrc to caps_vidconvsrc\n")
        return None
        
    pad = caps_vidconvsrc.get_static_pad("src")
    ghostpad = Gst.GhostPad.new("src", pad)
    bin_pad=nbin.add_pad(ghostpad)
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    
    return nbin
    
def make_element(element_name, i, post=False, rgba=False):
    """
    Creates a Gstreamer element with unique name
    Unique name is created by adding element type and index e.g. `element_name-i`
    Unique name is essential for all the element in pipeline otherwise gstreamer will throw exception.
    :param element_name: The name of the element to create
    :param i: the index of the element in the pipeline
    :return: A Gst.Element object
    """
    if post:
        unique_name = f"{element_name}-{i}-post"
    elif rgba:
        unique_name = f"{element_name}-{i}-rgba"
    else:
        unique_name = f"{element_name}-{i}"
    element = Gst.ElementFactory.make(element_name, unique_name)
    if not element:
        sys.stderr.write(" Unable to create {0}".format(element_name))
    # element.set_property("name", "{0}-{1}".format(element_name, str(i)))
    return element
    
    
def parse_args():

    parser = argparse.ArgumentParser(prog="deepstream_lpr", description="deepstream-test3 multi stream, multi model inference reference app")
    parser.add_argument(
        "-i",
        "--ids",
        help="IDS camera IDs or names",
        nargs="+",
        metavar="DEVICES",
        default=["IDS Imaging Development Systems GmbH-1409f4e6ad7d-4108758397"],
        required=False, 
    )
    
    parser.add_argument(
        "-u",
        "--usb",
        help="Path to usb webcam input streams (e.g., /dev/video0)",
        nargs="+",
        metavar="DEVICES",
        default=[],
        required=False, 
    )
    
    parser.add_argument(
        "-c",
        "--csi",
        help="Index of csi camera input streams",
        nargs="*",
        metavar="INDEX",
        type=int,
        default=[],
        required=False,
    )
    
    # Check input arguments
    '''
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        '''
    args = parser.parse_args()

    ids_stream_paths = args.ids
    usb_stream_paths = args.usb
    csi_stream_paths = args.csi
    return ids_stream_paths, usb_stream_paths, csi_stream_paths
    
def detect_plate(ids_args, usb_args, csi_args):
    args = ids_args + usb_args + csi_args
    print(args)
    global perf_data
    perf_data = PERF_DATA(len(args))

    number_sources = len(args)
    ids_number_sources = len(ids_args)
    usb_number_sources = len(usb_args)
    
    global folder_name
    folder_name = "out_crops"
    
    if not path.exists(folder_name):
        os.mkdir(folder_name)
    
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    
    print("Creating streamux \n ")
    
    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")
    
    pipeline.add(streammux)
    
    for i in range(number_sources):
        if not path.exists(folder_name + "/stream_" + str(i)):
            os.mkdir(folder_name + "/stream_" + str(i))
        print("Creating source_bin ",i," \n ")
        uri_name=args[i]
        if i >= ids_number_sources + usb_number_sources:
            source_bin = create_csi_source_bin(i, uri_name)
        elif i >= ids_number_sources:
            source_bin = create_usb_source_bin(i, uri_name)
        else:
            source_bin = create_ids_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname="sink_%u" %i
        sinkpad= streammux.get_request_pad(padname) 
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad=source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)
        
    queue1 = Gst.ElementFactory.make("queue", "queue1")
    pipeline.add(queue1)
    
    print("Creating Pgie \n ")
    # Use nvinfer to run inferencing on decoder's output,
    # behaviour of inferencing is set through config file
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")
    
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv1")
    if not nvvidconv1:
        sys.stderr.write(" Unable to create nvvidconv1 \n")
        sys.exit(1)
    nvvidconv1.set_property("nvbuf-memory-type", 4)
    pipeline.add(nvvidconv1)
        
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    if not filter1:
        sys.stderr.write(" Unable to get the caps filter1 \n")
        sys.exit(1)
    filter1.set_property("caps", caps1)
    pipeline.add(filter1)

    sgie1 = Gst.ElementFactory.make("nvinfer", "secondary1-nvinference-engine")
    if not sgie1:
        sys.stderr.write(" Unable to make sgie1 \n")
    
    print("Creating nvstreamdemux \n ")
    nvstreamdemux = Gst.ElementFactory.make("nvstreamdemux", "nvstreamdemux")
    if not nvstreamdemux:
        sys.stderr.write(" Unable to create nvstreamdemux \n")
    
    if is_live:
        print("Atleast one of the sources is live")
        streammux.set_property("live-source", 1)
        
    streammux.set_property('width', 640)
    streammux.set_property('height', 480)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', 4000000)
    streammux.set_property('live-source', 1)
    streammux.set_property('enable-padding', 1)
    
    
    #Set properties of pgie and sgie
    pgie.set_property('config-file-path', "dslpr_multi_pgie_config.txt")
    sgie1.set_property('config-file-path', "dslpr_sgie1_config.txt")
    
    pgie_batch_size = pgie.get_property("batch-size")
    if(pgie_batch_size != number_sources):
        print("WARNING: Overriding infer-config batch-size", pgie_batch_size, " with number of sources ", number_sources," \n")
        pgie.set_property("batch-size", number_sources)

    #Set properties of tracker
    config = configparser.ConfigParser()
    config.read('dslpr_tracker_config.txt')
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width' :
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        if key == 'enable-batch-process' :
            tracker_enable_batch_process = config.getint('tracker', key)
            tracker.set_property('enable_batch_process', tracker_enable_batch_process)
        if key == 'enable-past-frame' :
            tracker_enable_past_frame = config.getint('tracker', key)
            tracker.set_property('enable_past_frame', tracker_enable_past_frame)
    
    print("Adding elements to Pipeline \n")
    
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(sgie1)
    pipeline.add(nvstreamdemux)
    
    # linking
    streammux.link(queue1)
    queue1.link(nvvidconv1)
    nvvidconv1.link(filter1)
    filter1.link(pgie)
    pgie.link(tracker)
    tracker.link(sgie1)
    sgie1.link(nvstreamdemux)

    ##creating demux src

    for i in range(number_sources):
        # pipeline nvstreamdemux -> queue -> nvvidconv -> nvosd -> (if Jetson) nvegltransform -> nveglgl
        # Creating EGLsink
        
        if is_aarch64():
            print("Creating nv3dsink \n")
            sink = make_element("nv3dsink", i)
            if not sink:
                sys.stderr.write(" Unable to create nv3dsink \n")
        else:
            print("Creating EGLSink \n")
            sink = make_element("nveglglessink", i)
            if not sink:
                sys.stderr.write(" Unable to create egl sink \n")
        pipeline.add(sink)
        
        nvvidconv1 = make_element("nvvideoconvert", i, rgba=True)
        if not nvvidconv1:
            sys.stderr.write(" Unable to create nvvidconv1 \n")
        pipeline.add(nvvidconv1)
        
        print("Creating filter1 \n ")
        caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
        filter1 = make_element("capsfilter", i, rgba=True)
        if not filter1:
            sys.stderr.write(" Unable to get the caps filter1 \n")
        filter1.set_property("caps", caps1)
        pipeline.add(filter1)

        # creating queue
        queue = make_element("queue", i)
        pipeline.add(queue)

        # creating nvvidconv
        nvvideoconvert = make_element("nvvideoconvert", i)
        pipeline.add(nvvideoconvert)

        # creating nvosd
        nvdsosd = make_element("nvdsosd", i)
        pipeline.add(nvdsosd)
        """
        nvvidconv_postosd = make_element("nvvideoconvert", i, True)
        pipeline.add(nvvidconv_postosd)
        
        capsfilter = make_element("capsfilter", i)

        caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")
        capsfilter.set_property("caps", caps)
        pipeline.add(capsfilter)

        print("Creating H264 Encoder")
        encoder = make_element("nvv4l2h264enc", i)
        encoder.set_property("bitrate", 4000000)
        encoder.set_property('preset-level', 1)
        encoder.set_property('insert-sps-pps', 1)
        pipeline.add(encoder)

        # Make the payload-encode video into RTP packets
        print("Creating H264 rtppay")
        rtppay = make_element("rtph264pay", i)
        pipeline.add(rtppay)
        
        # Make the UDP sink
        updsink_port_num = 5400 + i
        print("Creating sink")
        sink = make_element("udpsink", i)
        sink.set_property('host', '224.224.255.255')
        sink.set_property('port', updsink_port_num)
        sink.set_property('async', False)
        sink.set_property('sync', 1)
        pipeline.add(sink)

        # connect nvstreamdemux -> queue
        padname = "src_%u" % i
        demuxsrcpad = nvstreamdemux.get_request_pad(padname)
        if not demuxsrcpad:
            sys.stderr.write("Unable to create demux src pad \n")

        queuesinkpad = queue.get_static_pad("sink")
        if not queuesinkpad:
            sys.stderr.write("Unable to create queue sink pad \n")
        demuxsrcpad.link(queuesinkpad)


        # connect  queue -> nvvidconv -> nvosd -> nveglgl
        queue.link(nvvideoconvert)
        nvvideoconvert.link(nvdsosd)
        nvdsosd.link(nvvidconv_postosd)
        nvvidconv_postosd.link(capsfilter)
        capsfilter.link(encoder)
        encoder.link(rtppay)
        rtppay.link(sink)
        """
        # connect nvstreamdemux -> queue
        padname = "src_%u" % i
        demuxsrcpad = nvstreamdemux.get_request_pad(padname)
        if not demuxsrcpad:
            sys.stderr.write("Unable to create demux src pad \n")

        queuesinkpad = queue.get_static_pad("sink")
        if not queuesinkpad:
            sys.stderr.write("Unable to create queue sink pad \n")
        demuxsrcpad.link(queuesinkpad)


        # connect  queue -> nvvidconv -> nvosd -> nveglgl
        queue.link(nvvidconv1)
        nvvidconv1.link(filter1)
        filter1.link(nvvideoconvert)
        nvvideoconvert.link(nvdsosd)
        nvdsosd.link(sink)
        
        filter1srcpad = filter1.get_static_pad("src")
        if not filter1srcpad:
            sys.stderr.write(" Unable to get src pad of filter1 \n")
        filter1srcpad.add_probe(Gst.PadProbeType.BUFFER, filter1_src_pad_buffer_probe, 0)
        
        sink.set_property("qos", 0)
        sink.set_property('sync', 0)
        #sink.set_property('async', 0)
        
        
    print("Linking elements in the Pipeline \n")
    
    """
    # Start multiple RTSP servers for each stream
    rtsp_port_base = 8554
    updsink_port_base = 5400
    codec = "H264"

    for i in range(number_sources):
        rtsp_port_num = rtsp_port_base + i
        updsink_port_num = updsink_port_base + i
        create_rtsp_server(rtsp_port_num, updsink_port_num, codec, i)
    """
    # print("Current working directory: ", os.getcwd())
    # output_file = os.path.join(os.getcwd(), "rtsp_pipeline.dot")
    # Gst.debug_bin_to_dot_file(pipeline, Gst.DebugGraphDetails.ALL, "rtsp_pipeline")
    # print(f".dot file created at: {output_file}")
    
    # create and event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)
    

    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    
    streamdemuxsinkpad = nvstreamdemux.get_static_pad("sink")
    if not streamdemuxsinkpad:
        sys.stderr.write(" Unable to get sink pad of streamdemux \n")
    streamdemuxsinkpad.add_probe(Gst.PadProbeType.BUFFER, streamdemux_sink_pad_buffer_probe, 0)
    
    
    # List the sources
    print("Now playing...")
    for i, source in enumerate(args):
        print(i, ": ", source)
        
    print("Starting pipeline \n")
    start = time.time()
    # start play back and listed to events
    pipeline.set_state(Gst.State.PLAYING)
    try:
      loop.run()
    except:
      pass
    
    # cleanup
    pipeline.set_state(Gst.State.NULL)
    
    end = time.time() - start
    print(end)

if __name__ == '__main__':
    ids_stream_paths, usb_stream_paths, csi_stream_paths = parse_args()
    sys.exit(detect_plate(ids_stream_paths, usb_stream_paths, csi_stream_paths))

