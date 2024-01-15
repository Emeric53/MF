import matplotlib.pyplot as plt

wavelength = []
value = []

with open("EMIT_unit_absorption_spectrum.txt", "r") as file:
    data = file.readlines()
for i in range(len(data)):
    data[i] = data[i].split(" ")
    value.append(float(data[i][1]))
    wavelength.append(float(data[i][0]))
plt.plot(wavelength, value)
plt.show()
