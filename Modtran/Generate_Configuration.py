import numpy as np

# 自定义一个浓度为1900ppb的甲烷廓线的大气模型,用于透射率模拟
def scale_methane_profile():
    enhance_range = np.array([1900])
    # 其他参数已经设置好的基础配置文件,用于修改廓线.
    file_name = r"C:\\PcModWin5\\Usr\\AHSI_trans.ltn" 
    for i in enhance_range:
        with open(file_name, 'r') as input_file:
            alllines = input_file.readlines()
            # 获取自定义大气廓线行的数据
            lines = alllines[7:160]
            #基于 大气层数进行内容修改
            for index in range(0, 51):
                # 逐层获取高程和对应甲烷混合比
                elevation = altitude[index]
                methane_concentration = methane[index]
                # 按位置进行填充 利用字符串的格式化确保填充后的长度一致
                lines[3*index] = f'{elevation:10f}' + lines[0][10:]
                lines[3*index+1] = lines[1][:20]+f'{methane_concentration*i/1900:10f}'+lines[1][30:]
                lines[3*index+2] = lines[2]
            # 将修改后的内容写入新的文件
            with open(f"C:\\PcModWin5\\Usr\\AHSI_trans_1900.ltn", 'w') as output_file:
                former_lines = alllines[:7]
                for former_line in former_lines:
                    output_file.write(former_line)
                for line in lines:
                    output_file.write(line)
                latter_lines = alllines[160:]
                for latter_line in latter_lines:
                    output_file.write(latter_line)


def add_8km_methane():
    enhance_range = np.array([5000,7500,10000])
    filename = r"C:\\PcModWin5\\Usr\\AHSI_1900.ltn" 
    for i in enhance_range:
        with open(filename, 'r') as input_file:
            alllines = input_file.readlines()
            # 获取自定义大气廓线行的数据
            lines = alllines[7:161]
            #基于 大气层数进行内容修改
            for index in range(0, 10):
                lines[3*index+1] = lines[3*index+1][:20]+f'{float(lines[3*index+1][20:30])+i*0.000125:10f}'+lines[3*index+1][30:]
            # 将修改后的内容写入新的文件
            with open(f"C:\\PcModWin5\\Usr\\AHSI_1900_interval_{int(i)}ppmm.ltn", 'w') as output_file:
                former_lines = alllines[:7]
                for former_line in former_lines:
                    output_file.write(former_line)
                for line in lines:
                    output_file.write(line)
                latter_lines = alllines[161:]
                for latter_line in latter_lines:
                    output_file.write(latter_line)

def add_500m_methane():
    enhance_range = np.arange(0,40.1,0.1)
    interval_range = np.array([5000,7500,10000])
    for interval in interval_range:
        for i in enhance_range:
            file_name = f"C:\\PcModWin5\\Usr\\AHSI_1900_interval_{int(interval)}ppmm.ltn" 
            with open(file_name, 'r') as input_file:
                alllines = input_file.readlines()
                # 获取自定义大气廓线行的数据
                line = alllines[8]
                line = line[:20]+f'{float(line[20:30])+i:10f}'+line[30:]                
                # 将修改后的内容写入新的文件
                with open(f"C:\\PcModWin5\\Usr\\AHSI_Methane_{int(i*500)}_ppmm_interval{int(interval)}.ltn", 'w') as output_file:
                    former_lines = alllines[:8]
                    for former_line in former_lines:
                        output_file.write(former_line)
                    output_file.write(line)
                    latter_lines = alllines[9:]
                    for latter_line in latter_lines:
                        output_file.write(latter_line)
                    
if __name__ == "__main__":
    # altitude = np.load('./Needed_data/altitude_profile.npy')
    # methane = np.load('./Needed_data/midlat_summer_1900ppm.npy')
    # scale_methane_profile()
    # add_8km_methane()
    # add_500m_methane()
    with open(r"C:\\PcModWin5\\Bin\\pcmodwin_batch.txt",'w') as batch:
        enhance_range = np.arange(0,40.1,0.1)
        interval_range = np.array([7500,10000])
        for interval in interval_range:
            for i in enhance_range:
                batch.write(f"C:\\PcModWin5\\Usr\\AHSI_Methane_{int(i*500)}_ppmm_interval{int(interval)}.ltn"+"\n")

