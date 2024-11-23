import numpy as np

a = ".\\data\\uas_files\\AHSI_UAS_0.txt"
with open(a, "r") as f:
    data = f.readlines()
    data = [i.strip() for i in data]
    data = [i.split(",") for i in data]
    data = np.array(data)
    print(data)
