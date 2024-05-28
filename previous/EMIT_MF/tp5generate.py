# 循环生成100个新文件,格式为 tp5  用于不同情况甲烷浓度增强的模拟
import math

for n in range(8):
    # 打开原始文件进行读取
    with open(r"C:\Users\RS\Desktop\EMIT_0.tp5", 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 修改第九行和第十一行的内容
    for line_number in [8, 11]:  # 第九行和第十二行对应的索引分别为8和11
        line = lines[line_number]
        original_line = ' '+line.strip()  # 去除行尾的换行符
        original_number = float(original_line[20:29])  # 获取原始数字
        modified_number = original_number + 2*math.pow(2,n)  # 修改数字
        modified_line = original_line[:20] + f'{modified_number:10f}' + original_line[30:]  # 使用f-string格式化科学计数法的数字，并确保占够10个字符
        lines[line_number] = modified_line + '\n'  # 更新行内容
    # 将修改后的内容写入modtran运行文件夹
    file_name = f"C:\\Users\\RS\\Desktop\\EMIT_tp5\\Emit_{int(math.pow(2,n)*1000)}.tp5"
    with open(file_name, 'w', encoding='utf-8') as file:
        file.writelines(lines)
