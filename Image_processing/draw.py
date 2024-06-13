import numpy as np
import matplotlib.pyplot as plt

with open(r"C:\PcModWin5\Bin\tape7.scn", "r") as transmittance_file:
    transmittance_data = transmittance_file.readlines()[11:-2]
    wavelength = [1/float(data.strip().split(' ')[0])*10000000 for data in transmittance_data]
    methane_transmittance = [float(data[115:122]) for data in transmittance_data]
    total_transmittance = [float(data[13:23]) for data in transmittance_data]


plt.plot(wavelength, methane_transmittance,color='blue')
plt.plot(wavelength, total_transmittance, color='black')
# plt.scatter(wavelength,methane_transmittance, c='r', s=1.5)
# plt.scatter(wavelength,total_transmittance, c='r', s=1.5)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmittance")
# plt.stem(wavelength, np.ones_like(wavelength))

# # Create a stem plot
# markerline, stemlines, baseline = plt.stem(wavelength,np.ones_like(wavelength), linefmt='r-', markerfmt=' ', basefmt=' ')
#
# # Set the linewidth of the stemlines
# plt.setp(stemlines, 'linewidth', 0.05)
plt.legend(['Methane_Transmittance','Total_Transmittance'], loc='lower right')
plt.title("Transmittance of Methane in SWIR")
plt.savefig('transmittance.png')
plt.xlim(1500,2500)
plt.show()


wavelength= np.array(wavelength)
transmittance_data = np.array(methane_transmittance)
total = np.array(total_transmittance)
np.save('wavelength.npy', wavelength)
np.save('transmittance.npy', transmittance_data)
np.save("total_transmittance.npy",total)
