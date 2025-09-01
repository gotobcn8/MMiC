import logbook
from logbook import Logger, FileHandler
import os

# Define the path for the log file
log_file_path = os.path.join('logs', 'example.log')

# Create a FileHandler instance
file_handler = FileHandler(log_file_path, mode='a', bubble=True)

# Create a logger instance and add the file handler
log = Logger('Example Logger')
log.handlers.append(file_handler)

# Log messages with different levels
log.trace('This is a trace message')
log.debug('This is a debug message')
log.info('This is an info message')
log.warning('This is a warning message')
log.error('This is an error message')
log.critical('This is a critical message')