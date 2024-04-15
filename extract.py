import os

# 设置解压文件路径 和 导出位置
folder = "J:\高光谱数据1"
filefolder = os.listdir("J:\高光谱数据1")
target_folder = "F:\\ahsi"
target_folder2 = "H:\\AHSI_part2"
existing_files = set(file for file in os.listdir(target_folder) + os.listdir(target_folder2))


# 遍历当前目录中的所有文件, 解压以.rar结尾的文件
for file in filefolder:
    # 判断文件是否为 .tar.xz 文件
    if file.endswith(".tar"):
        if str(file.rstrip('.tar.xz')) in existing_files:
            print("该文件已存在。")
        # 使用 Bandzip 解压文件
        else:
            file_path = os.path.join(folder, file)
            target_path = os.path.join(target_folder2, file.rstrip('.tar'))
            # open the file by the file_path and unzip it into target_folder
            os.system(f"E:\Bandizip\Bandizip.exe  x  -o:{target_folder2} {file_path}")


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
            print("该文件已存在。")
        # 使用 Bandzip 解压文件
        else:
            file_path = os.path.join(folder, file)
            # open the file by the file_path and unzip it into the same folder with the file
            os.system(f"E:\Bandizip\Bandizip.exe  x {file_path}")
