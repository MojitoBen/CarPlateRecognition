# log_config.py
import logging
import os
from datetime import datetime

def get_location():
    return '憲政路辦公室'

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


