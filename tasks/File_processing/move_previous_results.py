import os
import shutil
# This script is used to move the result of the image processing to the corresponding folder
# and export the RGB image.


def move_previous_result():
    target_path = "I:\\AHSI_result"
    filefolder_list = [
        "F:\\AHSI_part1",
        "H:\\AHSI_part2",
        "L:\\AHSI_part3",
        "I:\\AHSI_part4",
    ]
    for filefolder in filefolder_list:
        filelist, namelist = get_subdirectories(filefolder)
        # 遍历每一个数据文件夹 并进行处理
        for index in range(len(filelist)):
            # 获取 SW波段的文件路径
            filepath = os.path.join(
                filelist[index] + "\\result\\" + namelist[index] + "_SW.tif"
            )
            rpbpath = os.path.join(
                filelist[index] + "\\result\\" + namelist[index] + "_SW.rpb"
            )
            if os.path.exists(filepath):
                shutil.copy(filepath, target_path)
            if os.path.exists(rpbpath):
                shutil.copy(rpbpath, target_path)
            print(filepath + " is finished")


def move_rpbs():
    target_path = "I:\\AHSI_result"
    filefolder_list = [
        "F:\\AHSI_part1",
        "H:\\AHSI_part2",
        "L:\\AHSI_part3",
        "I:\\AHSI_part4",
    ]
    for filefolder in filefolder_list:
        filelist, namelist = get_subdirectories(filefolder)
        # 遍历每一个数据文件夹 并进行处理
        for index in range(len(filelist)):
            # 获取 SW波段的文件路径
            rpbpath = os.path.join(filelist[index], namelist[index] + "_SW.rpb")
            if os.path.exists(rpbpath):
                shutil.copy(rpbpath, target_path)
            print(rpbpath + " is finished")


def get_subdirectories(folder_path):
    """
    获取指定文件夹中所有子文件夹的路径列表。
    :param folder_path: 父文件夹的路径。
    :return: 子文件夹路径列表。
    """
    subdirectories = [
        os.path.join(folder_path, name)
        for name in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, name))
    ]
    filename = [
        name
        for name in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, name))
    ]
    return subdirectories, filename
