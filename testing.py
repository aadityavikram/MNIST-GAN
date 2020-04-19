import sys
import logging

root = logging.getLogger()
root.setLevel(logging.INFO)

stdout = logging.StreamHandler(sys.stdout)
stdout.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s -%(filename)s - %(levelname)s - %(message)s')
stdout.setFormatter(formatter)
root.addHandler(stdout)

logging.info('Hey there')
