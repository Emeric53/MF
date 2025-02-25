import os
import shutil

# 设置路径
selection_folder = (
    r"/media/emeric/Leap/EMIT_L2B_CH4PLM_shanxi"  # 目标文件夹（包含原始文件）
)
source_folder = "/media/emeric/Documents/shanxi_emit"  # 包含需要挑选文件名的文件路径
destination_folder = (
    r"/media/emeric/Documents/official_emit_shanxi"  # 复制到的目标文件夹路径
)

# 遍历挑选文件夹中的所有文件
for selected_filename in os.listdir(selection_folder):
    # 检查文件名是否符合挑选文件的格式
    if selected_filename.startswith("EMIT_L2B_CH4PLM"):
        # 提取挑选文件中的日期部分：假设日期在文件名的第24到第35个字符位置
        selected_date = selected_filename[20:35]

        # 遍历目标文件夹中的所有文件
        for target_filename in os.listdir(source_folder):
            # 检查目标文件是否符合目标文件格式
            if target_filename.startswith("EMIT_L1B_RAD"):
                # 提取目标文件中的日期部分：假设日期在文件名的第24到第35个字符位置
                target_date = target_filename[17:32]

                # 如果日期匹配
                if selected_date == target_date:
                    # 构造源文件和目标文件的完整路径
                    source_path = os.path.join(source_folder, target_filename)
                    destination_path = os.path.join(destination_folder, target_filename)

                    # 复制文件到目的文件夹
                    if os.path.exists(destination_path):
                        print(f"文件 {target_filename} 已存在于目标文件夹。")
                    else:
                        shutil.copy(source_path, destination_path)
                        print(f"文件 {target_filename} 已复制到目标文件夹。")

print("操作完成。")
