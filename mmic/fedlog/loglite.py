import logbook
from colorama import Fore, Style, init
 
# Initialize colorama
init(autoreset=True)
 
class ColorizedConsoleHandler(logbook.StreamHandler):
    def emit(self, record):
        msg = self.format(record)
        if record.level == logbook.DEBUG:
            msg = Fore.CYAN + msg
        elif record.level == logbook.INFO:
            msg = Fore.GREEN + msg
        elif record.level == logbook.WARNING:
            msg = Fore.YELLOW + msg
        elif record.level == logbook.ERROR:
            msg = Fore.RED + msg
        elif record.level == logbook.CRITICAL:
            msg = Fore.RED + Style.BRIGHT + msg
        print(msg)
 
# Setup the logger
logbook.set_datetime_format("local")
logger = logbook.Logger('My App')
handler = ColorizedConsoleHandler()
handler.push_application()
 
# Example log messages
logger.debug('This is a debug message')
logger.info('This is an info message')
logger.warning('This is a warning message')
logger.error('This is an error message')
logger.critical('This is a critical message')