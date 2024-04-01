import matplotlib.pyplot as plt

wavelengthlist = []
datalist = []
with open("unit_absorption_spectrum.txt",'r')as file:
    data = file.readlines()
    for line in data:
        print(line)
        wvl = float(line.split(' ')[0])
        value = float(line.split(' ')[1])
        wavelengthlist.append(wvl)
        datalist.append(value)
print(wavelengthlist)
print(datalist)
plt.plot(wavelengthlist,datalist)
plt.xlim(2100,2500)
plt.show()