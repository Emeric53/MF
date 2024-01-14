import math
import matplotlib.pyplot as plt
import numpy
import numpy as np

total_result = []
delta_crosssection = 1.8814446180099485e-21

with open("C:\\Users\\RS\\Desktop\\simu0.9.txt","r") as simu1:
    simu1data = simu1.readlines()[8:]
    wavelength_list = []
    radiancelist1 = []
    for i in simu1data:
        wavelength = float(i[2:12])
        wavelength_list.append(wavelength)
        radiance_1 = i[20:32]
        radiance_1 = float(radiance_1)*10000
        radiancelist1.append(radiance_1)
wlarray = np.array(wavelength_list)
index = np.where(wlarray == 2372.4939)
concentraion = math.log(radiancelist1[60]/radiancelist1[59], math.e)/delta_crosssection
print(concentraion)
total_result.append(concentraion)
with open("C:\\Users\\RS\\Desktop\\simu1.txt","r") as simu1:
    simu1data = simu1.readlines()[8:]
    wavelength_list = []
    radiancelist1 = []
    for i in simu1data:
        wavelength = float(i[2:12])
        wavelength_list.append(wavelength)
        radiance_1 = i[20:32]
        radiance_1 = float(radiance_1)*10000
        radiancelist1.append(radiance_1)
wlarray = np.array(wavelength_list)
index = np.where(wlarray == 2372.4939)
concentraion = math.log(radiancelist1[60]/radiancelist1[59], math.e)/delta_crosssection
print(concentraion)
total_result.append(concentraion)

with open("C:\\Users\\RS\\Desktop\\simu1.25.txt","r") as simu1:
    simu1data = simu1.readlines()[8:]
    wavelength_list = []
    radiancelist1 = []
    for i in simu1data:
        wavelength = float(i[2:12])
        wavelength_list.append(wavelength)
        radiance_1 = i[20:32]
        radiance_1 = float(radiance_1)*10000
        radiancelist1.append(radiance_1)
concentraion = math.log(radiancelist1[60]/radiancelist1[59], math.e)/delta_crosssection
print(concentraion)
total_result.append(concentraion)
with open("C:\\Users\\RS\\Desktop\\simu1.5.txt","r") as simu1:
    simu1data = simu1.readlines()[8:]
    wavelength_list = []
    radiancelist1 = []
    for i in simu1data:
        wavelength = float(i[2:12])
        wavelength_list.append(wavelength)
        radiance_1 = i[20:32]
        radiance_1 = float(radiance_1)*10000
        radiancelist1.append(radiance_1)
concentraion = math.log(radiancelist1[60]/radiancelist1[59], math.e)/delta_crosssection
print(concentraion)
total_result.append(concentraion)
with open("C:\\Users\\RS\\Desktop\\simu1.75.txt","r") as simu1:
    simu1data = simu1.readlines()[8:]
    wavelength_list = []
    radiancelist1 = []
    for i in simu1data:
        wavelength = float(i[2:12])
        wavelength_list.append(wavelength)
        radiance_1 = i[20:32]
        radiance_1 = float(radiance_1)*10000
        radiancelist1.append(radiance_1)
concentraion = math.log(radiancelist1[60]/radiancelist1[59], math.e)/delta_crosssection
print(concentraion)
total_result.append(concentraion)

with open("C:\\Users\\RS\\Desktop\\simu2.txt","r") as simu1:
    simu1data = simu1.readlines()[8:]
    wavelength_list = []
    radiancelist1 = []
    for i in simu1data:
        wavelength = float(i[2:12])
        wavelength_list.append(wavelength)
        radiance_1 = i[20:32]
        radiance_1 = float(radiance_1)*10000
        radiancelist1.append(radiance_1)
concentraion = math.log(radiancelist1[60]/radiancelist1[59], math.e)/delta_crosssection
print(concentraion)
total_result.append(concentraion)

concentration = [1.67*0.9]+[1.67*(i*0.25+1) for i in range(5)]
print(concentration)
plt.plot(concentration,total_result)
plt.show()
