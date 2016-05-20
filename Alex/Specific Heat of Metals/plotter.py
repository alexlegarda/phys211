from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

def read_data(datafile):
    data = np.genfromtxt(datafile, delimiter=',', skiprows=1)
    return data
    
data = read_data(r'data/Ni_data.csv')

print data

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1 = plt.axes()
ax1.errorbar(data[:,0], data[:,1], xerr=0.0, yerr=data[:,2], fmt='k.', label = 'Data')
ax1.set_title('Nickel Sample')
ax1.set_xlabel('Temperature (K)')
ax1.set_ylabel('Molar Specific Heat Capacity (J/K)')
plt.show()
plt.savefig('Ni.pdf')