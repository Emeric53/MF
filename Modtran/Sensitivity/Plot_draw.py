# Based on the simulation of modtran to get the sensitivity of atmospheric factors within the 1600-2500nm
# in the EMIT band spectral resolution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# input is the Modtran simulation file concolved with the EMIT band spectral resolution
Original_path = r"C:\Users\RS\Desktop\channels.out"
methane_path = r"C:\Users\RS\Desktop\channels_methane.out"
vapor_path = r"C:\Users\RS\Desktop\channels_vapor.out"
co2_path = r"C:\Users\RS\Desktop\channels_co2.out"
o3_path = r"C:\Users\RS\Desktop\channels_o3.out"
n2o_path = r"C:\Users\RS\Desktop\channels_n2o.out"
albedo_path = r"C:\Users\RS\Desktop\channels_albedo.out"

def get_wvl_list(datapath):
    with open(datapath, 'r') as f:
        datalines = f.readlines()[5:]
    band_wvl = []
    for line in datalines:
        band_wvl.append(float(line[285:295]))
    return band_wvl

def get_data_list(datapath):
    with open(datapath, 'r') as f:
        datalines = f.readlines()[5:]
    radiance = []
    for line in datalines:
        radiance.append(float(line[60:73]))
    return radiance

wvl = get_wvl_list(Original_path)
orginal = get_data_list(Original_path)
methane = get_data_list(methane_path)
vapor = get_data_list(vapor_path)
co2 = get_data_list(co2_path)
o3 = get_data_list(o3_path)
n2o = get_data_list(n2o_path)
albedo = get_data_list(albedo_path)

# output is the plot of the CH4 transmittance data or radiance data
fig, axes = plt.subplots(3,2,figsize=(20, 20), dpi=200, sharex=True, sharey=True)
# set the plot in row 1 and column 1
axes[0, 0].plot(wvl, orginal, 'b',label='Original',)
axes[0, 0].plot(wvl, methane,'r', label='Methane')
axes[0, 0].set_title('Methane Sensitivity')
axes[0, 0].set_ylabel('Radiance')
# set the plot in row 1 and column 2
axes[0, 1].plot(wvl, orginal,'b', label='Original')
axes[0, 1].plot(wvl, vapor,'r', label='Vapor')
axes[0, 1].set_title('Vapor Sensitivity')
# set the plot in row 2 and column 1
axes[1, 0].plot(wvl, orginal,'b', label='Original')
axes[1, 0].plot(wvl, co2, 'r', label='Carbon Dioxide')
axes[1, 0].set_title('Carbon Dioxide Sensitivity')
# set the plot in row 2 and column 2
axes[1, 1].plot(wvl, orginal, 'b', label='Original')
axes[1, 1].plot(wvl, o3, 'r', label='O3')
axes[1, 1].set_title('O3 Sensitivity')
axes[1, 1].set_ylabel('Radiance')
# set the plot in row 3 and column 1
axes[2, 0].plot(wvl, orginal, 'b', label='Original')
axes[2, 0].plot(wvl, n2o, 'r', label='N2O')
axes[2, 0].set_title('N2O Sensitivity')
axes[2, 0].set_ylabel('Radiance')
axes[2, 0].set_xlabel('Wavelength (nm)')
# set the plot in row 3 and column 2
axes[2, 1].plot(wvl, orginal, 'b', label='Original')
axes[2, 1].plot(wvl, albedo, 'r', label='Albedo')
axes[2, 1].set_title('Albedo Sensitivity')
axes[2, 1].set_xlabel('Wavelength (nm)')
# set the total plot
plt.legend()
plt.tight_layout()
plt.show()


