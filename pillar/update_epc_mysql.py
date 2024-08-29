import mysql.connector
from datetime import datetime, timedelta
import requests
'''
def setup_database():
    conn = mysql.connector.connect(
        host='35.187.153.63',
        port='3306',
        user='yunning',
        password='!Yunning90430522',
        database='yunning_pillar'
    )
    cursor = conn.cursor()
    conn.commit()
    return conn, cursor
'''
def setup_database():
    conn = mysql.connector.connect(
        host='127.0.0.1',
        port='3306',
        user='yunning',
        password='!Yunning90430522',
        database='yunning_pillar'
    )
    cursor = conn.cursor()
    conn.commit()
    return conn, cursor
'''
def insert_plate_num(cursor, plate, location):q
    #yolov8_pillar row490
    created_time = datetime.now()
    try:
        url = "http://35.187.153.63/insert_plate/"
        data = {
            "plate": f"{plate}",
            "location": f"{location}",
            "created_time": f"{created_time}",
            "last_time": f"{created_time}"
        }
        headers = {
            "Content-Type": "application/json"
        }
        print("data", data)
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        print(response.json())
    except requests.exceptions.RequestException as err:
        print("Insert Plate Number Failed:", err)

def insert_epc(cursor, epc, tid, location):
    created_time = datetime.now()
    try:
        url = "http://35.187.153.63/insert_epc/"
        data = {
            "code_number": f"{epc}",
            "TID": f"{tid}",
            "location": f"{location}",
            "created_time": f"{created_time}",
            "last_time": f"{created_time}"
        }
        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        print(response.json())
    except requests.exceptions.RequestException as err:
        print("Insert Etag Number Failed:", err)

def update_epc_last_seen_time(cursor, epc, tid, location, current_time): 
    try:
        url = "http://35.187.153.63/update_epc/"
        data = {
            "last_time": f"{current_time}",
            "code_number": f"{epc}",
            "TID": f"{tid}",
            "location": f"{location}"
        }
        headers = {
            "Content-Type": "application/json"
        }

        response = requests.put(url, json=data, headers=headers)
        response.raise_for_status()
        print(response.json())
    except requests.exceptions.RequestException as err:
        print("UPDATE Etag Number Last Time Failed:", err)

def update_plate_last_seen_time(cursor, plate, location, current_time):
    #yolov8_pillar row476
    try:
        url = "http://35.187.153.63/update_plate/"
        data = {
            "last_time": f"{current_time}",
            "plate": f"{plate}",
            "location": f"{location}"
        }
        headers = {
            "Content-Type": "application/json"
        }

        response = requests.put(url, json=data, headers=headers)
        response.raise_for_status()
        print(response.json())
    except requests.exceptions.RequestException as err:
        print("UPDATE Plate Number Last Time Failed:", err)
'''
'''
@router.post("/insert_plate/")
def create_plate_record_endpoint(record: PlateRecord):
    return create_plate_record(record)

@router.post("/insert_epc/")
def create_epc_record_endpoint(record: EPCRecord):
    return create_epc_record(record)

@router.put("/update_epc/")
def update_epc_endpoint(record: EPCRecord):
    return update_epc(record)

@router.put("/update_plate/")
def update_plate_endpoint(record: PlateRecord):
    return update_plate(record)
'''


#直接注入資料庫
#車牌&Etag相關
def insert_plate_num(cursor, plate, location):
    try:
        created_time = datetime.now()
        cursor.execute('INSERT INTO license_records (plate, location, created_time, last_time) VALUES (%s, %s, %s, %s)', (plate, location, created_time, created_time))
        if cursor.rowcount > 0:
            print(f"insert: {plate}, location: {location}.")
        else:
            print("Insert failed")
    except mysql.connector.Error as err:
        print(f"Error: {err}")

def insert_epc(cursor, epc, tid, location):
    try:
        created_time = datetime.now()
        cursor.execute('INSERT INTO sensors (code_number, TID, location, created_time, last_time) VALUES (%s, %s, %s, %s, %s)', (epc, tid, location, created_time, created_time))
        if cursor.rowcount > 0:
            print(f"insert : epc: {epc}, tid: {tid}, location: {location}.")
        else:
            print("Insert failed")
    except mysql.connector.Error as err:
        print(f"Error: {err}")

def update_epc_last_seen_time(cursor, epc, tid, location, current_time): 
    cursor.execute('UPDATE sensors SET last_time = %s WHERE code_number = %s AND TID = %s AND location = %s', (current_time, epc, tid, location))


def update_plate_last_seen_time(cursor, plate, location, current_time):
    cursor.execute('UPDATE license_records SET last_time = %s WHERE plate = %s AND location = %s', (current_time, plate, location))

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#配對相關

def insert_plate_sensor_mapping(cursor, plate, code_number, plate_time, sensor_time, current_time):
    cursor.execute("""
        INSERT INTO plate_sensor_mappings (plate, code_number, plate_time, sensor_time, created_time, last_time)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE last_time = %s;
    """, (plate, code_number, plate_time, sensor_time, current_time, current_time, current_time))

def insert_duplicates(duplicate_bindings, cursor, current_time):
    for plate, code_number in duplicate_bindings:
        cursor.execute("""
            INSERT INTO duplicate_plate_code_bindings (plate, code_number, found_time)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE found_time = %s;
        """, (plate, code_number, current_time, current_time))

def insert_unique_binding(cursor, plate, code_number, current_time):
    cursor.execute("""
        SELECT COUNT(*) FROM unique_plate_code_bindings
        WHERE plate = %s AND code_number = %s;
    """, (plate, code_number))
    exists = cursor.fetchone()[0] > 0

    if not exists:
        cursor.execute("""
            INSERT INTO unique_plate_code_bindings (plate, code_number, binding_time)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE binding_time = %s;
        """, (plate, code_number, current_time, current_time))

if __name__ == '__main__':
    conn, cursor = setup_database()
    print("conn", conn)

    created_time = datetime.now()
    epc = "ee0002220330000000028411"
    tid = "e2806894200050232a05813c"
    location = "憲政路辦公室"
    current_time = created_time

    #update_epc_last_seen_time(None, epc, tid, location, current_time)
    
    '''
    plate = "test123"
    location = "test123"
    '''
    #insert_plate_num(cursor, plate, location)
    #conn.commit()
    insert_epc(cursor, epc, tid, location)
    conn.commit()
