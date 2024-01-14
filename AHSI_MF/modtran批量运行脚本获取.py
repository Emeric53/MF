# 生成文件内容
file_content = [f"TEST/AHSI/AHSI_{n*100}.tp5\n" for n in range(101)]

# 将内容写入文件
with open('file_list.txt', 'w') as file:
    file.writelines(file_content)