from datetime import datetime
import const.constants as const

def getTime():
    datetime.now().strftime(const.TIME_FORMAT)