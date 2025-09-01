import logbook
from logbook import StreamHandler
import sys

# Create a StreamHandler to output logs to the console
StreamHandler(sys.stdout, bubble=True).push_application()

# Configure the formatting string for the info level
# to display the log messages in green color
formatter = logbook.StringFormatter(
    "{record.time} [{record.level_name:<8s}] {record.channel}: {record.message}",
    level_name={
        'INFO': lambda record, handler: '<fg#008000>{0}</>'.format(record.level_name),
        # Add custom colors for other levels if needed
    }
)
logbook.default_handler.formatter = formatter

# Log an info message
log = logbook.Logger('Example')
log.info('This is an info message')