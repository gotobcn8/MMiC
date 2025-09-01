from logbook import Logger, FileHandler, NOTSET
from logbook.more import ColorizedStderrHandler
from logbook._termcolors import colorize
import os
import sys
from logbook import(
    CRITICAL,
    DEBUG,
    ERROR,
    INFO,
    NOTICE,
    NOTSET,
    TRACE,
    WARNING,
)

route = 'fedlog/logfiles/client.log'
LOG_FILE_FORMAT = '({record.level_name})[{record.time:%Y-%m-%d %H:%M:%S}] {record.filename}:{record.lineno} {record.module}: {record.message}'

class ColoredStderrHandler(ColorizedStderrHandler):
    def get_color(self, record):
        """Returns the color for this record."""
        if record.level >= ERROR:
            return "red"
        elif record.level >= NOTICE:
            return "yellow"
        elif record.level >= INFO:
            return 'green'
        return "lightgray"
    def format(self, record):
        rv = super().format(record)
        if self.should_colorize(record):
            color = self.get_color(record)
            if color:
                rv = colorize(color, rv)
        return rv

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

glogger = Attender(index='global', filePath='global.log', handlers=[
    ColoredStderrHandler(
        bubble=True,
        format_string=LOG_FILE_FORMAT
    )
])

if __name__ == '__main__':
    glogger.info('test')
    glogger.warn('warn')
    glogger.error('error')
    glogger.critical('critical')