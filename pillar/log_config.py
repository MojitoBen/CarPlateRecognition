# log_config.py
import logging
import os
from datetime import datetime
import socket

location_dict = { '192.168.50.207': '憲政路辦公室', 
                  '127.0.0.1':'ben_mac_test',
                 'ip': 'xxx' } 

def get_location():
    ip_address = socket.gethostbyname(socket.gethostname())
    return location_dict.get(ip_address, ip_address)

def setup_logging(enable_logging=True):
    if enable_logging:
        log_directory = "logs"
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = os.path.join(log_directory, f'application_{current_time}.log')

        logging.basicConfig(filename=log_filename,
                            level=logging.INFO,
                            format='%(asctime)s %(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.CRITICAL)  # Effectively disable logging

if __name__ == '__main__': 
    location_t = get_location()
    print('地點:', location_t)