import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("C:\\Users\\RS\\Desktop\\[0]CH4,HITRAN_ X = 1.85, T = 297 K, P = 1 atm, L = 15 cm .csv")
wavelength = df[df.columns[0]].tolist()

print(1/wavelength*10000000)

# with open(,'r') as ahsi:
#     spectral = ahsi





