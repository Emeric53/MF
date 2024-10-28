import numpy as np

import os

altitude = np.load(
    "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\src\\data\\modtran_related\\altitude_profile.npy"
)
methane_profile = np.load(
    "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\src\\data\\modtran_related\\altitude_profile.npy"
)


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


# 调整配置文件中的数据以做到调整参数的目的
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


def generate_ltn_files_write_into_batchfile(
    input_file_path="C:\\PcModWin5\\Usr\\A_base.ltn",
    output_dir="C:\\PcModWin5\\Usr\\LUT\\",
    batch_file_path="C:\\PcModWin5\\Bin\\pcmodwin_batch.txt",
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
    修改甲烷增强值、SZA 和地表高度，生成相应的 .ltn 文件并将路径写入批处理文件。

    :param input_file_path: 输入文件的路径 (.ltn)
    :param output_dir: 输出文件的保存目录
    :param batch_file_path: 批处理文件的路径
    :param methane_step: 甲烷增强值的步长
    :param min_methane: 甲烷增强值的最小值
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

    # 打开批处理文件
    with open(batch_file_path, "w") as batch_file:
        # 遍历所有参数的组合
        for methane in methane_range:
            for sza in sza_range:
                for altitude in altitude_range:
                    # # 修改对应的行
                    # methane_value = methane / 500
                    # sza_value = sza
                    # altitude_value = altitude

                    # # 调用modify_line函数来修改相关行
                    # line8 = modify_line(
                    #     alllines[8],
                    #     20,
                    #     methane_value + float(alllines[8][20:30]),
                    #     30,
                    #     precision=6,
                    # )  # 修改第8行甲烷增强值
                    # line11 = modify_line(
                    #     alllines[11],
                    #     20,
                    #     methane_value + float(alllines[11][20:30]),
                    #     30,
                    #     precision=6,
                    # )  # 修改第11行甲烷增强值
                    # line166 = modify_line(
                    #     alllines[153], 10, sza_value, 20
                    # )  # 修改第166行 SZA
                    # line6 = modify_line(
                    #     alllines[5], 70, altitude_value, 80
                    # )  # 修改第6行地表高度
                    # line164 = modify_line(
                    #     alllines[151], 10, altitude_value, 20, precision=6
                    # )  # 修改第151行

                    # 生成输出文件名
                    output_file_name = os.path.join(
                        output_dir, f"{methane}_{sza}_{altitude}.ltn"
                    )

                    # 将路径写入批处理文件
                    batch_file.write(output_file_name + "\n")

                    # # 写入新的内容到输出文件
                    # with open(output_file_name, "w") as output_file:
                    #     # 写入未修改的前5行
                    #     output_file.writelines(alllines[:5])
                    #     # 写入修改后的第6行（地表高度）
                    #     output_file.write(line6)
                    #     output_file.writelines(alllines[6:8])
                    #     # 写入修改后的第8行（甲烷增强值）
                    #     output_file.write(line8)
                    #     output_file.writelines(alllines[9:11])
                    #     # 写入修改后的第11行（甲烷增强值）
                    #     output_file.write(line11)
                    #     output_file.writelines(alllines[12:151])
                    #     # 写入修改后的第164行（地表高度）
                    #     output_file.write(line164)
                    #     output_file.writelines(alllines[152:153])
                    #     # 写入修改后的第166行（SZA）
                    #     output_file.write(line166)
                    #     output_file.writelines(alllines[154:])


def generate_batch_file(
    result_dir="C:\\PcModWin5\\Bin\\batch",  # 结果文件夹路径
    input_dir="C:\\PcModWin5\\Usr\\LUT\\",  # 原始 .ltn 文件所在目录
    batch_file_path="C:\\PcModWin5\\Bin\\pcmodwin_batch.txt",  # 新生成的批处理文件名
    result_file_suffix="_tape7.txt",  # 结果文件的后缀
    size_threshold=400
    * 1024,  # 文件大小不足的阈值（700KB 转换为字节），如果文件小于此值，视为运行异常
):
    """
    根据结果文件的存在和大小生成需要重新运行的 .ltn 文件的批处理文件。

    :param result_dir: 结果文件的目录
    :param input_dir: 原始 .ltn 文件的目录
    :param batch_file_path: 生成的 .bat 文件路径
    :param result_file_suffix: 结果文件的后缀，例如 "_tape7.txt"
    :param size_threshold: 文件大小的阈值，低于此值的文件会被认为是异常的
    """
    # 获取输入目录下的所有 .ltn 文件
    ltn_files = [f for f in os.listdir(input_dir) if f.endswith(".ltn")]

    # 打开批处理文件进行写入
    with open(batch_file_path, "w") as batch_file:
        # 遍历每一个 .ltn 文件
        for ltn_file in ltn_files:
            # 构建对应的结果文件名
            base_name = os.path.splitext(ltn_file)[0]  # 获取文件名（无后缀）
            result_file = os.path.join(result_dir, f"{base_name}{result_file_suffix}")
            print(result_file)
            # 检查结果文件是否存在，以及文件大小是否大于阈值
            if (
                not os.path.exists(result_file)
                or os.path.getsize(result_file) < size_threshold
            ):
                # 如果文件不存在或者文件大小不足，写入原始 .ltn 文件路径到批处理文件中
                ltn_file_path = os.path.join(input_dir, ltn_file)
                batch_file.write(
                    f"{ltn_file_path}\n"
                )  # 假设使用 run_program.exe 运行 .ltn 文件

    print(f"Batch file '{batch_file_path}' generated successfully.")


if __name__ == "__main__":
    # generate_ltn_files_write_into_batchfile()
    generate_batch_file()
