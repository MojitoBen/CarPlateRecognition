import mysql.connector
from datetime import datetime, timedelta
import requests

created_time = datetime.now()

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

def insert_plate_num(cursor, plate, location):
    global created_time
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
    global created_time
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
    global created_time
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
    global created_time
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

'''
#直接注入資料庫
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
'''
if __name__ == '__main__':
    #conn, cursor = setup_database()
    #print("conn", conn)

    '''
    created_time = datetime.now()
    epc = "ee0002220330000000028411"
    tid = "e2806894200050232a05813c"
    location = "分局辦公室"
    current_time = created_time

    update_epc_last_seen_time(None, epc, tid, location, current_time)
    '''
    created_time = datetime.now()
    plate = "test123"
    location = "test123"
    current_time = created_time

    insert_plate_num(None, plate, location)
