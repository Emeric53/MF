# 生成文件内容
file_content = [f"TEST/EMIT/Emit_{n*100}.tp5\n" for n in range(101)]

# 将内容写入文件
with open("C:\\Users\\RS\\Desktop\\modtran5.2.6\\mod5root.in", 'w') as file:
    file.writelines(file_content)
