import os
import glob

# 指定目录路径
directory = "C:\\Users\\RS\\Desktop\\modtran5.2.6\\TEST\\SensitivityAnalysis"

# 获取目录中所有文件的名称
files = ["C:\\Users\\RS\\Desktop\\modtran5.2.6\\TEST\\SensitivityAnalysis\\"+os.path.basename(file) for file in glob.glob(os.path.join(directory, '*.tp5'))]

# 将文件名写入文本文件
with open("C:\\Users\\RS\\Desktop\\modtran5.2.6\\mod5root.in", 'w') as file:
    file.write('\n'.join(files))
