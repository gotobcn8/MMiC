import mmic.utils.read as read
import entrance
import mmic.utils.cmd.parse as parse
from gpu_mem_track import MemTracker
from mmic.fedlog.logbooker import glogger
default_config_path = 'config.yaml'

if __name__ == '__main__':
    parser = parse.get_cmd_parser().parse_args()
    # if parser.show:
    #     print()
    
    if parser.file == '':
        parser.file = default_config_path
    args = read.yaml_read(parser.file)
    gpu_tracker = MemTracker()  
    args['gpu_tracker'] = gpu_tracker
    glogger.debug(args)
    entrance.run(args)