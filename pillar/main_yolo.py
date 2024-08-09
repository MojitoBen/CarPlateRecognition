import logging
import os
from datetime import datetime, timedelta
from update_epc_mysql import *
from yolov8_pillar import *
from log_config import setup_logging, get_location
import threading
from time import sleep
from uhf.reader import *
import torch
#from deepstream_lpr_multi_ids_usb_csi import *

setup_logging(enable_logging=True)

def start_receiving_epc(conn, cursor, ip_address="192.168.0.169", port=8160):
    epc_dict = {}
    epc_time_dict = {}
    time_interval = timedelta(minutes=1) #如果出現後消失Ｎ分鐘後又出現則更新紀錄
    max_dict_size = 1000  # 設定字典的最大大小

    def receivedEpc(epcInfo: LogBaseEpcInfo):
        if epcInfo.result == 0:
            epc = epcInfo.epc
            tid = epcInfo.tid
            current_time = datetime.now()
            location = get_location()
            #print("epc_time_dict", epc_time_dict)

            if epc in epc_dict:
                last_seen_time = epc_time_dict[epc]
                if current_time - last_seen_time > time_interval:
                    logging.debug(f"EPC reappeared after interval: {epc}") # 超過時間間隔，視為新出現的 EPC
                    update_epc_last_seen_time(cursor, epc, tid, location, current_time)  # 更新最後出現時間
                    conn.commit()
                else:
                    logging.debug(f"EPC already seen, not updating last seen time: {epc}")
                epc_dict[epc] += 1
            else:
                if len(epc_dict) >= max_dict_size:
                    oldest_epc = next(iter(epc_dict))
                    del epc_dict[oldest_epc]
                    del epc_time_dict[oldest_epc]

                epc_dict[epc] = 1
                epc_time_dict[epc] = current_time
                logging.info(f"New EPC: {epc}")
                insert_epc(cursor, epc, tid, location)
                conn.commit()

    def receivedEpcOver(epcOver: LogBaseEpcOver):
        logging.info("LogBaseEpcOver")

    g_client = GClient()
    if g_client.openTcp((ip_address, port)):
        g_client.callEpcInfo = receivedEpc
        g_client.callEpcOver = receivedEpcOver

        msg = MsgBaseInventoryEpc(antennaEnable=EnumG.AntennaNo_1.value,
                                  inventoryMode=EnumG.InventoryMode_Inventory.value)
        tid = ParamEpcReadTid(mode=EnumG.ParamTidMode_Auto.value, dataLen=6)
        msg.readTid = tid
        if g_client.sendSynMsg(msg) == 0:
            logging.info(msg.rtMsg)

        try:
            while True:
                sleep(0.1)
        except KeyboardInterrupt:
            logging.info("EPC reception interrupted by user.")
        finally:
                stop = MsgBaseStop()
                if g_client.sendSynMsg(stop) == 0:
                    logging.info(stop.rtMsg)

                g_client.close()
                cursor.close()
                conn.close()
    else:
        logging.error("Failed to open RFID connection.")
        cursor.close()
        conn.close()

def start_detecting_plate(conn, cursor):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = LPRSystem.make_parser().parse_args()
    lpr_system = LPRSystem(args, device, plate_confidence_threshold=0.5, char_confidence_threshold=0.5)
    plate_info = lpr_system.detect_plate_in_video(conn, cursor, save=False, save_org=True) #save_org=True 存原影片
    logging.info("Plate detection completed.")
'''
def start_detecting(conn, cursor):
    ids_stream_paths, usb_stream_paths, csi_stream_paths = parse_args()
    detect_plate(ids_stream_paths, usb_stream_paths, csi_stream_paths)
    logging.info("Plate detection completed.")
'''
def connect_to_database():
    conn, cursor = setup_database()
    if conn and cursor:
        logging.info("MySQL connected.")
        return conn, cursor
    else:
        logging.error("Failed to connect to MySQL.")
        return None, None

def start_receiving_epc_thread():
    conn, cursor = connect_to_database()
    if conn and cursor:
        start_receiving_epc(conn, cursor, ip_address="192.168.0.169", port=8160)  # rfid接收器ip，可用程式調整
    else:
        logging.error("Failed to start EPC receiving thread due to database connection issue.")


def start_detecting_plate_thread():
    conn, cursor = connect_to_database()
    if conn and cursor:
        start_detecting_plate(conn, cursor)
        #start_detecting(conn, cursor)        
    else:
        logging.error("Failed to start plate detecting thread due to database connection issue.")


if __name__ == '__main__':
    epc_thread = threading.Thread(target=start_receiving_epc_thread)
    plate_thread = threading.Thread(target=start_detecting_plate_thread)

    epc_thread.start()
    plate_thread.start()

    epc_thread.join()
    plate_thread.join()
