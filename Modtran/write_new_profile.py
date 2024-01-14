"""该代码用于批量生成用于modtran模拟的配置文件，目前改变的变量是 甲烷的廓线"""
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
for i in range(101):  # 创建100个文件，可以根据需要修改数量
    file_name = f"data\\EMIT_{i * 100}.ltn"  # 定义文件名，例如：file_1.txt, file_2.txt, ...
    with open("C:\\PcModWin5\\Usr\\EMIT.ltn", 'r') as input_file:
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
                    ch4 += 0.2 * i
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
