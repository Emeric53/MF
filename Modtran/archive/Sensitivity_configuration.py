"""该代码用于生成 用于modtran模拟的tp5文件，改变参数为某一气体的含量"""
import os
name1 = ['n2o', 'co', 'ch4', 'o2', 'no', 'so2', 'no2', 'nh3', 'hno3', 'n2']
name2 = ['cfc11', 'cfc12', 'cfc13', 'cfc14', 'cfc22', 'cfc113', 'cfc114', 'cfc115', 'ciono2', 'hno4', 'chci2f', 'cci4', 'n2o5']
name3 = ['oh', 'hf', 'hcl', 'hbr', 'hi', 'clo', 'ocs', 'h2co', 'hoci', 'n2_2', 'hcn', 'ch3ci', 'h2o2', 'c2h2', 'c2h6', 'ph3']

# 读取原始tp5文件
with open("C:\\Users\\RS\\Desktop\\SensitivityAnalysis\\original.tp5", "r") as original:
    data = original.readlines()
    for index in range(len(name1)):
        output_file = f'C:\\Users\\RS\\Desktop\\SensitivityAnalysis\\{name1[index]}.tp5'
        if not os.path.exists(output_file):
            print("1")
        elif os.path.getsize(output_file) > 0:
            with open(output_file, 'w') as file:
                file.truncate(0)
        for i, line in enumerate(data):
            line = line.strip("\n")
            if i == 3:
                modified_line = ''
                modified_line = line.replace('1.000', '2.000', index+1)
                modified_line = modified_line.replace('2.000', '1.000', index)
                with open(output_file, 'a') as file:
                    file.write(modified_line+"\n")
                continue
            with open(output_file, 'a') as file:
                file.write(line+"\n")
