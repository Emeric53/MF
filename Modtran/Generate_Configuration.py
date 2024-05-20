"""
该代码用于批量生成用于modtran模拟的配置文件，目前改变的变量是 甲烷的廓线
"""
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

with open('US_1987_CH4_profile.txt', 'r') as file:
    data = file.readlines()
    for i in data:
        i = float(i.strip())
        methane.append(i)

file_name = r"C:\PcModWin5\Usr\EMIT.ltn"  # 定义文件名， file_1.txt, file_2.txt, ...
with open(file_name, 'r') as input_file:
    alllines = input_file.readlines()
    # 获取自定义大气廓线行的数据
    lines = alllines[7:157]
    #基于 大气层数进行内容修改
    for index in range(1, 50):
        # 逐层获取高程和对应甲烷混合比
        elevation = altitude[index]
        methane_concentration = methane[index]
        # 按位置进行填充 利用字符串的格式化确保填充后的长度一致
        lines[3*index] = f'{elevation:10f}' + lines[0][10:]
        lines[3*index+1] = lines[1][:20]+f'{methane_concentration:10f}'+lines[1][30:]
        lines[3*index+2] = lines[2]
    # 将修改后的内容写入新的文件
    with open("C:\PcModWin5\Usr\EMIT_test.ltn", 'w') as output_file:
        former_lines = alllines[:7]
        for former_line in former_lines:
            output_file.write(former_line)
        for line in lines:
            output_file.write(line)
        latter_lines = alllines[157:]
        for latter_line in latter_lines:
            output_file.write(latter_line)

#
# with open(file_name, 'r') as input_file:
#     with open(file_name, 'w') as temp_file:
#         lines = input_file.readlines()
#         lines[6] = "   51" + lines[6][5:]
#         for J in range(7):
#             temp_file.write(lines[J])
#         for level in range(51):
#             al = altitude[level]
#             ch4 = methane[level]
#             print(ch4)
#             if level == 0:
#                 ch4 += 2*math.pow(2,i)
#             print(ch4)
#             num1 = len(f"{al:.6f}")
#             lines[7] = " " * (10 - num1) + f"{al:.6f}" + lines[7][10:]
#             temp_file.write(lines[7])
#             num2 = len(f"{methane[0]:.3E}")
#             lines[8] = lines[8][:20] + " " * (10 - num2) + f"{ch4:.3E}" + lines[8][30:]
#             temp_file.write(lines[8])
#             temp_file.write(lines[9])
#             # 将原始文件中的剩余行写入
#         for line in lines[10:]:
#             temp_file.write(line)
#
# directory = r"E:\\modtran5.2.6\\TEST\\EMIT_tp5"
#
# # 获取目录中所有文件的名称
# files = [r"E:\\modtran5.2.6\\TEST\\EMIT_tp5\\"+os.path.basename(file) for file in glob.glob(os.path.join(directory, '*.tp5'))]
#
# # 将文件名写入文本文件
# with open(r"E:\modtran5.2.6\mod5root.in", 'w') as file:
#     file.write('\n'.join(files))
