import os
import shutil


def initialize_keras():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def remove_folder(del_dir):
    del_list = os.listdir(del_dir)

    for f in del_list:
        file_path = os.path.join(del_dir, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path, True)


def file_list(dir_path):
    list_file = os.listdir(dir_path)
    for i in range(0, len(list_file)):
        list_file[i] = dir_path + "/" + list_file[i]
    return list_file
