import numpy as np
import os
import sys

sys.path.append("C:\\Users\\RS\\VSCode\\matchedfiltermethod")

altitude = np.load("./MyData/altitude_profile.npy")
methane_profile = np.load("./MyData/midlat_summer_1900ppm.npy")


# 将一个modtran文件模板中的甲烷廓线进行缩放至想要的浓度
def scale_methane_profile():
    enhance_range = np.array([1900])
    # 其他参数已经设置好的基础配置文件,用于修改廓线.
    original_file_name = r"C:\\PcModWin5\\Usr\\AHSI_trans.ltn"
    for i in enhance_range:
        with open(original_file_name, "r") as input_file:
            alllines = input_file.readlines()
            # 获取自定义大气廓线行的数据
            lines = alllines[7:160]
            # 基于 大气层数进行内容修改
            for index in range(0, 51):
                # 逐层获取高程和对应甲烷混合比
                elevation = altitude[index]
                methane_concentration = methane_profile[index]
                # 按位置进行填充 利用字符串的格式化确保填充后的长度一致
                lines[3 * index] = f"{elevation:10f}" + lines[0][10:]
                lines[3 * index + 1] = (
                    lines[1][:20]
                    + f"{methane_concentration*i/1900:10f}"
                    + lines[1][30:]
                )
                lines[3 * index + 2] = lines[2]
            # 将修改后的内容写入新的文件
            with open(
                f"C:\\PcModWin5\\Usr\\AHSI_trans_{int(i)}.ltn", "w"
            ) as output_file:
                former_lines = alllines[:7]
                for former_line in former_lines:
                    output_file.write(former_line)

                for line in lines:
                    output_file.write(line)
                latter_lines = alllines[160:]

                for latter_line in latter_lines:
                    output_file.write(latter_line)


# 在8km范围内为甲烷廓线添加特定浓度的增强
def add_8km_methane():
    enhance_range = np.array([5000, 7500, 10000])
    filename = r"C:\\PcModWin5\\Usr\\AHSI_Methane_0_ppmm.ltn"
    for i in enhance_range:
        with open(filename, "r") as input_file:
            alllines = input_file.readlines()
            # 获取自定义大气廓线行的数据
            lines = alllines[7:161]
            # 基于 大气层数进行内容修改
            for index in range(0, 10):
                lines[3 * index + 1] = (
                    lines[3 * index + 1][:20]
                    + f"{float(lines[3*index+1][20:30])+i*0.000125:10f}"
                    + lines[3 * index + 1][30:]
                )
            # 将修改后的内容写入新的文件
            with open(
                f"C:\\PcModWin5\\Usr\\AHSI_Methane_1900ppmm_interval_{int(i)}ppmm.ltn",
                "w",
            ) as output_file:
                former_lines = alllines[:7]
                for former_line in former_lines:
                    output_file.write(former_line)
                for line in lines:
                    output_file.write(line)
                latter_lines = alllines[161:]
                for latter_line in latter_lines:
                    output_file.write(latter_line)


# 模拟地表0-500m的甲烷浓度增强
def methane_enhancements_interval_increase():
    enhance_range = np.arange(0, 50500, 500)
    # for interval in interval_range:
    for i in enhance_range:
        file_name = "C:\\PcModWin5\\Usr\\AHSI_1900ppb\\AHSI_Methane_0_ppmm.ltn"
        with open(file_name, "r") as input_file:
            alllines = input_file.readlines()
            # 获取自定义大气廓线行的数据
            line = alllines[8]
            line = line[:20] + f"{float(line[20:30])+i/500:10f}" + line[30:]
            line2 = alllines[11]
            line2 = line2[:20] + f"{float(line2[20:30])+i/500:10f}" + line2[30:]
            # 将修改后的内容写入新的文件
            with open(f"C:\\PcModWin5\\Usr\\LUT\\{i}_700_90.ltn", "w") as output_file:
                former_lines = alllines[:8]
                for former_line in former_lines:
                    output_file.write(former_line)

                output_file.write(line)

                middle_lines = alllines[9:11]
                for middle_line in middle_lines:
                    output_file.write(middle_line)

                output_file.write(line2)

                latter_lines = alllines[12:]
                for latter_line in latter_lines:
                    output_file.write(latter_line)


def new_methane_enhancements_interval_increase(
    input_file_path="C:\\PcModWin5\\Usr\\AHSI_1900ppb\\AHSI_Methane_0_ppmm.ltn",
    output_dir="C:\\PcModWin5\\Usr\\LUT\\",
    enhance_step=500,
    max_enhancement=50500,
):
    """
    Modify methane enhancement value and save new files at each interval.

    :param input_file_path: Path to the input .ltn file
    :param output_dir: Directory to save the modified output files
    :param enhance_step: Step size for methane enhancement
    :param max_enhancement: Maximum enhancement value
    """

    def modify_line(line: str, new_value: float) -> str:
        """
        Modify a line by updating the methane enhancement value.

        :param line: Original line to modify
        :param new_value: The new methane enhancement value to insert
        :return: Modified line as a string
        """
        # Assuming the methane value is located at position [20:30]
        # Modify the substring and return the new line
        return line[:20] + f"{new_value:10.6f}" + line[30:]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the entire input file once to avoid multiple file I/O
    with open(input_file_path, "r") as input_file:
        alllines = input_file.readlines()

    # Loop through each enhancement value
    enhance_range = np.arange(0, max_enhancement, enhance_step)

    for i in enhance_range:
        # Modify specific lines with the enhancement value
        new_value = i / 500
        line8 = modify_line(alllines[8], new_value)
        line11 = modify_line(alllines[11], new_value)

        # Generate output file name
        output_file_name = os.path.join(output_dir, f"{i}_700_90.ltn")

        # Write modified content to the new file
        with open(output_file_name, "w") as output_file:
            # Write the unmodified first 8 lines
            output_file.writelines(alllines[:8])
            # Write the modified 8th and 11th lines
            output_file.write(line8)
            output_file.writelines(alllines[9:11])
            output_file.write(line11)
            # Write the remaining unmodified lines
            output_file.writelines(alllines[12:])


def sza_interval_increase():
    enhance_range = np.arange(-3000, 50500, 500)
    # for interval in interval_range:
    for i in enhance_range:
        file_name = "C:\\PcModWin5\\Usr\\AHSI_1900ppb\\AHSI_Methane_0_ppmm.ltn"
        with open(file_name, "r") as input_file:
            alllines = input_file.readlines()
            # 获取自定义大气廓线行的数据
            line = alllines[8]
            line = line[:20] + f"{float(line[20:30])+i/500:10f}" + line[30:]
            line2 = alllines[11]
            line2 = line2[:20] + f"{float(line2[20:30])+i/500:10f}" + line2[30:]
            # 将修改后的内容写入新的文件
            with open(f"C:\\PcModWin5\\Usr\\LUT\\0_700_{i}.ltn", "w") as output_file:
                former_lines = alllines[:8]
                for former_line in former_lines:
                    output_file.write(former_line)

                output_file.write(line)

                middle_lines = alllines[9:11]
                for middle_line in middle_lines:
                    output_file.write(middle_line)

                output_file.write(line2)

                latter_lines = alllines[12:]
                for latter_line in latter_lines:
                    output_file.write(latter_line)


def sensor_altitude_interval_increase():
    interval_range = np.arange(-3000, 50500, 500)
    # for interval in interval_range:
    for sensor_altitude in interval_range:
        file_name = "C:\\PcModWin5\\Usr\\AHSI_1900ppb\\AHSI_Methane_0_ppmm.ltn"
        with open(file_name, "r") as input_file:
            alllines = input_file.readlines()
            # 获取自定义大气廓线行的数据
            line = alllines[8]
            line = line[:20] + f"{float(line[20:30])+sensor_altitude:10f}" + line[30:]
            line2 = alllines[11]
            line2 = (
                line2[:20]
                + f"{float(line2[20:30])+sensor_altitude/500:10f}"
                + line2[30:]
            )
            # 将修改后的内容写入新的文件
            with open(
                f"C:\\PcModWin5\\Usr\\LUT\\0_{sensor_altitude}_90.ltn", "w"
            ) as output_file:
                former_lines = alllines[:8]
                for former_line in former_lines:
                    output_file.write(former_line)

                output_file.write(line)

                middle_lines = alllines[9:11]
                for middle_line in middle_lines:
                    output_file.write(middle_line)

                output_file.write(line2)

                latter_lines = alllines[12:]
                for latter_line in latter_lines:
                    output_file.write(latter_line)
    for sensor_altitude in interval_range:
        with open("C:\\PcModWin5\\Bin\\pcmodwin_batch.txt", "w") as batch:
            batch.write(f"C:\\PcModWin5\\Usr\\LUT\\0_{sensor_altitude}_90.ltn" + "\n")


# 将配置文件路径写入modtran的批处理文件中
def write_batchfile():
    with open(r"C:\\PcModWin5\\Bin\\pcmodwin_batch.txt", "w") as batch:
        enhance_range = np.arange(-3000, 50500, 500)
        for i in enhance_range:
            batch.write(
                f"C:\\PcModWin5\\Usr\\AHSI_1900ppb\\AHSI_Methane_{int(i)}_ppmm.ltn"
                + "\n"
            )


if __name__ == "__main__":
    methane_enhancements_interval_increase()
    sza_interval_increase()
    sensor_altitude_interval_increase()
    # scale_methane_profile()
    # add_8km_methane()
    # add_500m_methane()
    # write_batchfile()
