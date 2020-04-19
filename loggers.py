import logging
import sys


__author__ = "dhruv"


# root = logging.getLogger('root') todo check why this doesn't work sometimes
root = logging.getLogger()
root.setLevel(logging.INFO)

stdout = logging.StreamHandler(sys.stdout)
stdout.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s -%(filename)s - %(levelname)s - %(message)s')
stdout.setFormatter(formatter)
root.addHandler(stdout)

stderr = logging.StreamHandler(sys.stderr)
stderr.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
stderr.setFormatter(formatter)
root.addHandler(stderr)


class Loggers:
    def __init__(self):
        pass

    @staticmethod
    def info(message):
        logging.info(message)

    @staticmethod
    def error(message):
        logging.error(message)

    @staticmethod
    def exception(message):
        logging.exception(message)
