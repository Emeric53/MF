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


def new_sza_interval_increase(
    input_file_path="C:\\PcModWin5\\Usr\\A_base.ltn",
    output_dir="C:\\PcModWin5\\Usr\\LUT\\",
    step=5,
    min_sza=0,
    max_sza=90,
):
    """
    Modify methane enhancement value and save new files at each interval.

    :param input_file_path: Path to the input .ltn file
    :param output_dir: Directory to save the modified output files
    :param step: Step size for methane enhancement
    :param max_sza: Maximum enhancement value
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
        return line[:10] + f"{new_value:10.3f}" + line[20:]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the entire input file once to avoid multiple file I/O
    with open(input_file_path, "r") as input_file:
        alllines = input_file.readlines()

    # Loop through each enhancement value
    enhance_range = np.arange(min_sza, max_sza + step, step)

    for i in enhance_range:
        # Modify specific lines with the enhancement value
        new_value = i
        line166 = modify_line(alllines[165], new_value)

        # Generate output file name
        output_file_name = os.path.join(output_dir, f"0_700_{i}.ltn")

        # Write modified content to the new file
        with open(output_file_name, "w") as output_file:
            # Write the unmodified first 8 lines
            output_file.writelines(alllines[:165])
            # Write the modified 166th lines
            output_file.write(line166)
            output_file.writelines(alllines[166:])


def new_surface_altitude_interval_increase(
    input_file_path="C:\\PcModWin5\\Usr\\A_base.ltn",
    output_dir="C:\\PcModWin5\\Usr\\LUT\\",
    step=1,
    min_sza=0,
    max_sza=5,
):
    """
    Modify methane enhancement value and save new files at each interval.

    :param input_file_path: Path to the input .ltn file
    :param output_dir: Directory to save the modified output files
    :param step: Step size for methane enhancement
    :param max_sza: Maximum enhancement value
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
        return line[:70] + f"{new_value:10.3f}" + line[80:]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the entire input file once to avoid multiple file I/O
    with open(input_file_path, "r") as input_file:
        alllines = input_file.readlines()

    # Loop through each enhancement value
    enhance_range = np.arange(min_sza, max_sza + step, step)

    for i in enhance_range:
        # Modify specific lines with the enhancement value
        new_value = i
        line6 = modify_line(alllines[5], new_value)

        # Generate output file name
        output_file_name = os.path.join(output_dir, f"0_{i}_0.ltn")

        # Write modified content to the new file
        with open(output_file_name, "w") as output_file:
            # Write the unmodified first 8 lines
            output_file.writelines(alllines[:5])
            # Write the modified 6th lines
            output_file.write(line6)
            output_file.writelines(alllines[6:])


def modify_line(
    line: str, start_idx: int, new_value: float, end_idx: int, precision=3
) -> str:
    """
    通用的行修改函数，插入新的数值到指定位置。

    :param line: 原始行
    :param start_idx: 数值开始的位置
    :param new_value: 新的数值
    :param end_idx: 数值结束的位置
    :param precision: 数值的精度（小数点位数）
    :return: 修改后的行
    """
    return line[:start_idx] + f"{new_value:10.{precision}f}" + line[end_idx:]


def generate_ltn_files(
    input_file_path="C:\\PcModWin5\\Usr\\A_base.ltn",
    output_dir="C:\\PcModWin5\\Usr\\LUT\\",
    methane_step=500,
    min_methane=0,
    max_methane=50000,
    sza_step=5,
    min_sza=0,
    max_sza=90,
    altitude_step=1,
    min_altitude=0,
    max_altitude=5,
):
    """
    修改甲烷增强值、SZA 和地表高度，生成相应的 .ltn 文件。

    :param input_file_path: 输入文件的路径 (.ltn)
    :param output_dir: 输出文件的保存目录
    :param methane_step: 甲烷增强值的步长
    :param max_methane: 甲烷增强值的最大值
    :param sza_step: SZA 步长
    :param min_sza: SZA 最小值
    :param max_sza: SZA 最大值
    :param altitude_step: 地表高度步长
    :param min_altitude: 地表高度最小值
    :param max_altitude: 地表高度最大值
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取输入文件内容
    with open(input_file_path, "r") as input_file:
        alllines = input_file.readlines()

    # 生成甲烷增强值、SZA 和地表高度的范围
    methane_range = np.arange(min_methane, max_methane + methane_step, methane_step)
    sza_range = np.arange(min_sza, max_sza + sza_step, sza_step)
    altitude_range = np.arange(
        min_altitude, max_altitude + altitude_step, altitude_step
    )

    # 遍历所有参数的组合
    for methane in methane_range:
        for sza in sza_range:
            for altitude in altitude_range:
                # 修改对应的行
                methane_value = methane / 500
                sza_value = sza
                altitude_value = altitude

                line8 = modify_line(
                    alllines[8], 20, methane_value, 30, precision=6
                )  # 修改第8行甲烷增强值（6位小数）
                line11 = modify_line(
                    alllines[11], 20, methane_value, 30, precision=6
                )  # 修改第11行甲烷增强值（6位小数）
                line166 = modify_line(
                    alllines[165], 10, sza_value, 20
                )  # 修改第166行 SZA
                line6 = modify_line(
                    alllines[5], 70, altitude_value, 80
                )  # 修改第6行地表高度
                line164 = modify_line(
                    alllines[163], 10, altitude_value, 20, precision=6
                )
                # 生成输出文件名
                output_file_name = os.path.join(
                    output_dir, f"{methane}_{sza}_{altitude}.ltn"
                )

                # 写入新的内容到输出文件
                with open(output_file_name, "w") as output_file:
                    # 写入未修改的前5行
                    output_file.writelines(alllines[:5])
                    # 写入修改后的第6行（地表高度）
                    output_file.write(line6)
                    output_file.writelines(alllines[6:8])
                    # 写入修改后的第8行（甲烷增强值）
                    output_file.write(line8)
                    output_file.writelines(alllines[9:11])
                    # 写入修改后的第11行（甲烷增强值）
                    output_file.write(line11)
                    output_file.writelines(alllines[12:163])
                    # 写入修改后的第164行（地表高度）
                    output_file.write(line164)
                    output_file.writelines(alllines[164:165])
                    # 写入修改后的第166行（SZA）
                    output_file.write(line166)
                    output_file.writelines(alllines[166:])


# 将配置文件路径写入modtran的批处理文件中
def write_ltn_files_inbatchfile(
    methane_step=500,
    min_methane=0,
    max_methane=50000,
    sza_step=5,
    min_sza=0,
    max_sza=90,
    altitude_step=1,
    min_altitude=0,
    max_altitude=5,
):
    """
    修改甲烷增强值、SZA 和地表高度，生成相应的 .ltn 文件。

    :param input_file_path: 输入文件的路径 (.ltn)
    :param output_dir: 输出文件的保存目录
    :param methane_step: 甲烷增强值的步长
    :param max_methane: 甲烷增强值的最大值
    :param sza_step: SZA 步长
    :param min_sza: SZA 最小值
    :param max_sza: SZA 最大值
    :param altitude_step: 地表高度步长
    :param min_altitude: 地表高度最小值
    :param max_altitude: 地表高度最大值
    """

    # 生成甲烷增强值、SZA 和地表高度的范围
    methane_range = np.arange(min_methane, max_methane + methane_step, methane_step)
    sza_range = np.arange(min_sza, max_sza + sza_step, sza_step)
    altitude_range = np.arange(
        min_altitude, max_altitude + altitude_step, altitude_step
    )
    with open(r"C:\\PcModWin5\\Bin\\pcmodwin_batch.txt", "w") as batch:
        # 遍历所有参数的组合
        for methane in methane_range:
            for sza in sza_range:
                for altitude in altitude_range:
                    batch.write(
                        f"C:\\PcModWin5\\Usr\\LUT\\{methane}_{sza}_{altitude}.ltn"
                        + "\n"
                    )


if __name__ == "__main__":
    generate_ltn_files()
    write_ltn_files_inbatchfile()
    # scale_methane_profile()
    # add_8km_methane()
    # add_500m_methane()
    # write_batchfile()
