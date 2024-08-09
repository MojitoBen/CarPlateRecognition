import logging
from log_config import setup_logging, get_location
from update_epc_mysql import *
import threading
import time

setup_logging(enable_logging=True)

def handle_duplicate_bindings(conn, cursor, interval=30):
    try:
        # Get the current group_id
        cursor.execute("SELECT current_group FROM group_counter ORDER BY id DESC LIMIT 1;")
        current_group = cursor.fetchone()[0]

        processed_ids_plate = set()
        processed_ids_code = set()

        # Handle duplicates by plate
        cursor.execute("""
            SELECT id, plate, code_number, binding_time
            FROM unique_plate_code_bindings
            WHERE plate IN (
                SELECT plate
                FROM unique_plate_code_bindings
                GROUP BY plate
                HAVING COUNT(*) > 1
            );
        """)
        plate_duplicates = cursor.fetchall()

        if plate_duplicates:
            for duplicate in plate_duplicates:
                id, plate, code_number, binding_time = duplicate

                if id in processed_ids_plate:
                    continue

                # Check if this duplicate already belongs to a group with the correct count
                if group_exists(cursor, plate, count_duplicates(cursor, 'plate', plate)):
                    continue

                current_group += 1

                plate_targets = set()
                plate_stack = [(id, plate, code_number, binding_time)]
                count = 0
                while plate_stack:
                    current_id, current_plate, current_code_number, current_binding_time = plate_stack.pop()
                    if current_id in processed_ids_plate:
                        continue

                    processed_ids_plate.add(current_id)
                    plate_targets.add(current_plate)
                    count += 1

                    cursor.execute("""
                        INSERT INTO duplicate_bindings (binding_id, plate, code_number, binding_time, group_id, notes)
                        VALUES (%s, %s, %s, %s, %s, %s);
                    """, (current_id, current_plate, current_code_number, current_binding_time, current_group, ''))
                    conn.commit()

                    cursor.execute("""
                        SELECT id, plate, code_number, binding_time
                        FROM unique_plate_code_bindings
                        WHERE plate = %s AND id != %s;
                    """, (current_plate, current_id))
                    related_duplicates = cursor.fetchall()

                    for related_duplicate in related_duplicates:
                        if related_duplicate[0] not in processed_ids_plate:
                            plate_stack.append(related_duplicate)

                if plate_targets:
                    targets = list(plate_targets)[0]
                    group_id = insert_or_update_group(cursor, current_group, targets, count)

                    # Insert or update duplicate_bindings for plate
                    for current_id, current_plate, current_code_number, current_binding_time in plate_stack:
                        insert_if_not_exists(cursor, conn, current_id, current_plate, current_code_number, current_binding_time, group_id)


        # Handle duplicates by code_number
        cursor.execute("""
            SELECT id, plate, code_number, binding_time
            FROM unique_plate_code_bindings
            WHERE code_number IN (
                SELECT code_number
                FROM unique_plate_code_bindings
                GROUP BY code_number
                HAVING COUNT(*) > 1
            );
        """)
        code_duplicates = cursor.fetchall()

        if code_duplicates:
            for duplicate in code_duplicates:
                id, plate, code_number, binding_time = duplicate

                if id in processed_ids_code:
                    continue

                # Check if this duplicate already belongs to a group with the correct count
                if group_exists(cursor, code_number, count_duplicates(cursor, 'code_number', code_number)):
                    continue

                current_group += 1

                code_targets = set()
                code_stack = [(id, plate, code_number, binding_time)]
                count = 0
                while code_stack:
                    current_id, current_plate, current_code_number, current_binding_time = code_stack.pop()
                    if current_id in processed_ids_code:
                        continue

                    processed_ids_code.add(current_id)
                    code_targets.add(current_code_number)
                    count += 1

                    cursor.execute("""
                        INSERT INTO duplicate_bindings (binding_id, plate, code_number, binding_time, group_id, notes)
                        VALUES (%s, %s, %s, %s, %s, %s);
                    """, (current_id, current_plate, current_code_number, current_binding_time, current_group, ''))
                    conn.commit()

                    cursor.execute("""
                        SELECT id, plate, code_number, binding_time
                        FROM unique_plate_code_bindings
                        WHERE code_number = %s AND id != %s;
                    """, (current_code_number, current_id))
                    related_duplicates = cursor.fetchall()

                    for related_duplicate in related_duplicates:
                        if related_duplicate[0] not in processed_ids_code:
                            code_stack.append(related_duplicate)

                if code_targets:
                    targets = list(code_targets)[0]
                    group_id = insert_or_update_group(cursor, current_group, targets, count)

                    # Insert or update duplicate_bindings for code_number
                    for current_id, current_plate, current_code_number, current_binding_time in code_stack:
                        insert_if_not_exists(cursor, conn, current_id, current_plate, current_code_number, current_binding_time, group_id)

        # Update the current group number
        cursor.execute("""
            UPDATE group_counter
            SET current_group = %s
            ORDER BY id DESC
            LIMIT 1;
        """, (current_group,))
        conn.commit()

        logging.info("Duplicate bindings handled.")
    except Exception as e:
        logging.error(f"Failed to handle duplicate bindings: {e}")
    
    time.sleep(interval)

def group_exists(cursor, target, count):
    cursor.execute("""
        SELECT group_id, record_count
        FROM duplicate_groups
        WHERE targets = %s;
    """, (target,))
    existing_group = cursor.fetchone()

    if existing_group:
        existing_group_id, existing_count = existing_group
        if existing_count == count:
            return True
    return False

def count_duplicates(cursor, column, value):
    cursor.execute(f"""
        SELECT COUNT(*)
        FROM unique_plate_code_bindings
        WHERE {column} = %s;
    """, (value,))
    return cursor.fetchone()[0]

def insert_or_update_group(cursor, group_id, targets, count):
    cursor.execute("""
        SELECT group_id, record_count
        FROM duplicate_groups
        WHERE targets = %s;
    """, (targets,))
    existing_group = cursor.fetchone()

    if existing_group:
        existing_group_id, existing_count = existing_group

        if count > existing_count:
            # Update existing group with new count and reset flags
            cursor.execute("""
                UPDATE duplicate_groups
                SET record_count = %s, checked = %s, solved = %s, noted = %s
                WHERE group_id = %s;
            """, (count, False, False, '有新增重複資料', existing_group_id))
            logging.info(f"group: {existing_group_id}, 有新增 {targets} 的重複資料")
        return existing_group_id
    else:
        # Insert new group
        cursor.execute("""
            INSERT INTO duplicate_groups (group_id, targets, record_count, checked, solved, noted)
            VALUES (%s, %s, %s, %s, %s, %s);
        """, (group_id, targets, count, False, False, False))
        return group_id
    
def insert_if_not_exists(cursor, conn, current_id, current_plate, current_code_number, current_binding_time, group_id):
    cursor.execute("""
        SELECT 1
        FROM duplicate_bindings
        WHERE binding_id = %s;
    """, (current_id,))
    if not cursor.fetchone():
        cursor.execute("""
            INSERT INTO duplicate_bindings (binding_id, plate, code_number, binding_time, group_id, notes)
            VALUES (%s, %s, %s, %s, %s, %s);
        """, (current_id, current_plate, current_code_number, current_binding_time, group_id, ''))
        conn.commit()

def update_plate_sensor_mappings(conn, cursor, interval=30):
    while True:
        try:
            current_time = datetime.now()
            time_window_start = current_time - timedelta(seconds=interval)
            time_window_end = current_time - timedelta(seconds=10)

            # 插入新的配對記錄，考慮時間窗口
            cursor.execute("""
                INSERT INTO plate_sensor_mappings (plate, code_number, plate_time, sensor_time, created_time, last_time)
                SELECT lr.plate, s.TID, lr.created_time, s.created_time, %s, %s
                FROM license_records lr
                JOIN sensors s
                ON ABS(TIMESTAMPDIFF(SECOND, lr.created_time, s.created_time)) <= 3
                AND lr.created_time BETWEEN %s AND %s
                AND s.created_time BETWEEN %s AND %s
                AND NOT EXISTS (
                    SELECT 1
                    FROM plate_sensor_mappings psm
                    WHERE psm.plate = lr.plate AND psm.code_number = s.TID
                    AND psm.plate_time = lr.created_time AND psm.sensor_time = s.created_time
                );
            """, (current_time, current_time, time_window_start, time_window_end, time_window_start, time_window_end))
            conn.commit()

            # 找出在時間範圍內唯一配對的車牌和傳感器代碼
            cursor.execute("""
                SELECT psm.plate, psm.code_number
                FROM plate_sensor_mappings psm
                JOIN (
                    SELECT plate, code_number, COUNT(*) as cnt
                    FROM plate_sensor_mappings
                    GROUP BY plate, code_number
                    HAVING cnt = 1
                ) as unique_psm
                ON psm.plate = unique_psm.plate AND psm.code_number = unique_psm.code_number;
            """)
            unique_mappings = cursor.fetchall()

            for plate, code_number in unique_mappings:
                # Check for duplicate entries for plate or code_number
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM unique_plate_code_bindings
                    WHERE plate = %s OR code_number = %s;
                """, (plate, code_number))
                count = cursor.fetchone()[0]

                if count < 2:  # Ensure there isn't a record where both are repeated simultaneously
                    current_time = datetime.now()
                    cursor.execute("""
                        INSERT INTO unique_plate_code_bindings (plate, code_number, binding_time)
                        VALUES (%s, %s, %s)
                        ON DUPLICATE KEY UPDATE binding_time = %s;
                    """, (plate, code_number, current_time, current_time))
                    conn.commit()

            # 檢查並更新重複的車牌和傳感器代碼
            cursor.execute("""
                SELECT plate, code_number
                FROM (
                    SELECT plate, code_number
                    FROM plate_sensor_mappings
                    WHERE (plate, code_number) IN (
                        SELECT plate, code_number
                        FROM plate_sensor_mappings
                        GROUP BY plate, code_number
                        HAVING COUNT(*) > 1
                    )
                    ORDER BY sensor_time DESC
                    LIMIT 1
                ) as latest_mapping;
            """)
            latest_mapping = cursor.fetchone()

            if latest_mapping:
                plate, code_number = latest_mapping
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM unique_plate_code_bindings
                    WHERE plate = %s OR code_number = %s;
                """, (plate, code_number))
                count = cursor.fetchone()[0]

                if count < 2:  # Ensure there isn't a record where both are repeated simultaneously
                    current_time = datetime.now()
                    cursor.execute("""
                        INSERT INTO unique_plate_code_bindings (plate, code_number, binding_time)
                        VALUES (%s, %s, %s)
                        ON DUPLICATE KEY UPDATE binding_time = %s;
                    """, (plate, code_number, current_time, current_time))
                    conn.commit()

            logging.info("Plate-sensor mappings updated.")
        except Exception as e:
            logging.error(f"Failed to update plate-sensor mappings: {e}")

        time.sleep(interval)

def insert_data(conn, cursor):
    # Simulate inserting data for testing purposes
    data = [
        ('AXY-8888', '000000000000202405210009', '2024-06-12 14:29:35'),
        ('CCC-9931', '000000000000202405210009', '2024-06-18 14:29:35'),
        ('CCC-9931', '777700000000202405210009', '2024-06-24 14:29:35')
    ]

    insert_query = """
    INSERT INTO unique_plate_code_bindings (plate, code_number, binding_time)
    VALUES (%s, %s, %s);
    """

    cursor.executemany(insert_query, data)
    conn.commit()

def start_handle_duplicate_bindings_thread():
    conn, cursor = setup_database()
    if conn and cursor:
        handle_duplicate_bindings(conn, cursor, interval=30)
    else:
        logging.error("Failed to start duplicate bindings thread due to database connection issue.")

def start_update_mappings_thread():
    conn, cursor = setup_database()
    if conn and cursor:
        update_plate_sensor_mappings(conn, cursor, interval=30)
    else:
        logging.error("Failed to start update mappings thread due to database connection issue.")

if __name__ == '__main__':

    handle_duplicate_bindings_thread = threading.Thread(target=start_handle_duplicate_bindings_thread)
    update_mappings_thread = threading.Thread(target=start_update_mappings_thread)

    handle_duplicate_bindings_thread.start()
    update_mappings_thread.start()

    handle_duplicate_bindings_thread.join()
    update_mappings_thread.join()
