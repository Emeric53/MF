"""该代码用于批量生成用于modtran模拟的配置文件，目前改变的变量是 甲烷的廓线"""
import math
import os
import glob

#读取 海拔 和 对应层甲烷浓度
altitude = []
methane = []
with open('altitude.txt', 'r') as file:
    data = file.readlines()
    for i in data:
        i = float(i.strip())
        altitude.append(i)

with open('1.85_US_1987_CH4_profile.txt', 'r') as file:
    data = file.readlines()
    for i in data:
        i = float(i.strip())
        methane.append(i)

# 基于需求 对配置文件进行修改

# 批量修改并生成 ltn文件
for i in range(8):
    file_name = f"C:\\Users\\RS\\Desktop\\EMIT_cfg\\EMIT_{int(math.pow(2,i) * 1000)}.ltn"  # 定义文件名， file_1.txt, file_2.txt, ...
    with open(r"C:\\Users\RS\\Desktop\\EMIT_0.ltn", 'r') as input_file:
        with open(file_name, 'w') as temp_file:
            lines = input_file.readlines()
            lines[6] = "   51" + lines[6][5:]
            for J in range(7):
                temp_file.write(lines[J])
            for level in range(51):
                al = altitude[level]
                ch4 = methane[level]
                print(ch4)
                if level == 0 or level == 1:
                    ch4 += 2*math.pow(2,i)
                print(ch4)
                num1 = len(f"{al:.6f}")
                lines[7] = " " * (10 - num1) + f"{al:.6f}" + lines[7][10:]
                temp_file.write(lines[7])
                num2 = len(f"{methane[0]:.3E}")
                lines[8] = lines[8][:20] + " " * (10 - num2) + f"{ch4:.3E}" + lines[8][30:]
                temp_file.write(lines[8])
                temp_file.write(lines[9])
                # 将原始文件中的剩余行写入
            for line in lines[10:]:
                temp_file.write(line)

# 将生成的tp5文件路径写入 modtran 批量处理文件中
# 指定目录路径
directory = r"E:\\modtran5.2.6\\TEST\\EMIT_tp5"

# 获取目录中所有文件的名称
files = [r"E:\\modtran5.2.6\\TEST\\EMIT_tp5\\"+os.path.basename(file) for file in glob.glob(os.path.join(directory, '*.tp5'))]

# 将文件名写入文本文件
with open(r"E:\modtran5.2.6\mod5root.in", 'w') as file:
    file.write('\n'.join(files))
