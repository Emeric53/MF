# 用于批量文件的解压缩 以及删除已经解压的文件

import os

# 设置解压文件路径 和 导出位置
# folder = "J:\\高光谱数据1"
# filefolder = os.listdir("J:\\高光谱数据1")
folder = "J:\高光谱数据2"
filefolder = os.listdir("J:\\高光谱数据2")
target_folder1 = "F:\\AHSI_part1"
target_folder2 = "H:\\AHSI_part2"
target_folder3 = "L:\\AHSI_part3"
target_folder4 = "I:\\AHSI_part4"
existing_files = set(file for file in os.listdir(target_folder1) + os.listdir(target_folder2) + os.listdir(target_folder3)+os.listdir(target_folder4))
print(len(existing_files))

# 遍历当前目录中的所有文件, 解压以.rar结尾的文件
for file in filefolder:
    # 判断文件是否为 .tar.xz 文件
    if file.endswith(".tar"):
        if str(file.rstrip('.tar')) in existing_files:
            print("该.tar文件已存在。")
        # 使用 Bandzip 解压文件
        else:
            file_path = os.path.join(folder, file)
            target_path = os.path.join(target_folder3, file.rstrip('.tar'))
            # open the file by the file_path and unzip it into target_folder
            os.system(f"E:\Bandizip\\bc.exe x -o:{target_folder4} {file_path}")
            print(file_path+' 解压完成')

# 删除已经解压的文件 减少不必要的体积占用
for filename in filefolder:
        if filename.endswith('.tar'):
            if str(filename.rstrip('.tar')) in existing_files:
                # 构建文件的完整路径
                file_path = os.path.join(folder, filename)
                # 删除文件
                os.remove(file_path)
                print(f"已删除文件: {file_path}")

#遍历当前目录中的所有文件，解压以.tar.xz结尾的文件
for file in filefolder:
    # 判断文件是否为 .tar.xz 文件
    if file.endswith(".tar.xz"):
        if str(file.rstrip('.tar.xz')) in existing_files:
            pass
        else:
            file_path = os.path.join(folder, file)
            # open the file by the file_path and unzip it into the same folder with the file
            os.system(f"E:\\Bandizip\\bc.exe x -o:{folder} {file_path}")
            print(file_path+' 解压完成')
