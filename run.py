import subprocess
import mmic.utils.cmd.parse as parse
from datetime import datetime
import os

def run_with_dir(parser):
    config_files = os.listdir(parser.file)
    dir,base = check_log_path(parser.log)
    if base != '':
        print(f'Because you are running with directory, so we only generate files under {dir}')
    for config_file in config_files:
        if not (config_file.endswith('.yaml') or config_file.endswith('.yml')):
            continue
        print(config_file)
        logname = config_file.split('.')[0] + '-' + datetime.now().strftime("%y-%m-%d-%H-%M-%S") + '.log'
        execute(
            os.path.join(parser.file,config_file),
            os.path.join(dir,logname),
        )
    
def execute(config_file,log_file):
    # run program and output it
    with open(log_file, "w+") as lf:
        process = subprocess.Popen(
            ["nohup", "python", "main.py", "-f", config_file],
            stdout=lf,
            stderr=subprocess.STDOUT
        )
        # get process id
        lf.write(f"\nPID: {process.pid}\n")

    print(f"Started process run config {config_file} with PID & output to {log_file}: {process.pid}")


def check_log_path(log_path):
    directory = log_path
    basename = ''
    if directory.endswith('.log'):
        directory = os.path.dirname(directory)
        basename = os.path.basename(log_path)
    if not os.path.exists(directory):
        os.makedirs(directory,766)
    return directory,basename

if __name__ == '__main__':
    parser = parse.get_cmd_parser().parse_args()
    # print(parser.args)
    
    # Check config file
    if os.path.isdir(parser.file):
        response = input(f"Your config file input is a directory,do you want to run all the files under this {parser.file} ? (y):").strip()
        if response == 'y':
            run_with_dir(parser)
    else:
        dir,base = check_log_path(parser.log)
        if base == '':
            base = os.path.basename(parser.file).split('.')[0] + '.log'
        execute(
            parser.file,
            os.path.join(dir,base)
        )