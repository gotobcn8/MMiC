from logbook import Logger, FileHandler, NOTSET
from logbook.more import ColorizedStderrHandler
import sys
from colorama import init, Fore
# import const.constants as const

route = 'fedlog/logfiles/client.log'
LOG_FILE_FORMAT = '({record.level_name})[{record.time:%Y-%m-%d %H:%M:%S}] {record.filename}:{record.lineno} {record.module}: {record.message}'
init()
class ColoredStderrHandler(ColorizedStderrHandler):
    def get_color_formatter(self, record):
        if record.level_name == 'INFO':
            color = Fore.GREEN + '{0}' + Fore.RESET
        elif record.level_name == 'WARNING':
            color = Fore.YELLOW + '{0}' + Fore.RESET
        else:
            color = '{0}'
        return color.format(record.level_name)

class Attender(Logger):
    def __init__(self, index='client', filePath='file.log', level=NOTSET, handlers=None) -> None:
        super().__init__(index, level)
        self.handlers = [FileHandler(
            filename=filePath, format_string=LOG_FILE_FORMAT, bubble=True,
        )]
        if handlers is not None:
            if isinstance(handlers, list):
                self.handlers.extend(handlers)
            else:
                self.handlers.append(handlers)

# clogger = Attender(index='client')
# slogger = Attender(index='server')
# errlogger = Attender()
glogger = Attender(index='global', filePath='global.log', handlers=[
    ColoredStderrHandler(
        bubble=True,
        format_string=LOG_FILE_FORMAT
    )
])

if __name__ == '__main__':
    # chandler = ColorizedStderrHandler()
    # fhandler = TimedRotatingFileHandler(
    #     filename='fedlog/logfiles/client.log'
    # )
    # clogger = Attender(handlers=[chandler, fhandler])
    print("\033[92mThis should be green\033[0m")
    glogger.info('test')
    glogger.warn('warn')
    # clogger.info('this is a info log')
    # clogger.warn('this is a warn log')