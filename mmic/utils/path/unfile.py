import tarfile
import os

def extract_tar_gz(source_path, dest_path):
    # check target directory
    print(source_path,dest_path)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path,mode=755,exist_ok=True)
    
    # open tar.gz and untar it
    with tarfile.open(source_path, "r:gz") as tar:
        tar.extractall(path=dest_path)

# example
# source_file = 'your_archive.tar.gz'  # your tar.gz dir
# destination_dir = '/path/to/destination'  # your target dir

# extract_tar_gz(source_file, destination_dir)
