from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

def read_data(datafile):
    data = np.genfromtxt(datafile, delimiter=',', skiprows=1)
    return data
    
data = read_data(r'data/Ni_lin.csv')

t = data[:,0]*data[:,0]
y = 2*data[:,1]/data[:,0]
y_err = data[:,2]/data[:,1] * y

def linear(p, x):
    return p[0] * (x - p[1])
def residual(p, x, y, err):
    return (linear(p, x) - y) / err
    
p1 = [0.1, 0.]
pf1, cov1, info1, mesg1, success1 = optimize.leastsq(residual, p1, args=(t, y, y_err), full_output=1)
chisq1 = sum(info1["fvec"]*info1["fvec"])
dof1 = len(t)-len(pf1)
pferr1 = [np.sqrt(cov1[i,i]) for i in range(len(pf1))]



fig = plt.figure()
ax = fig.add_subplot(111)
ax = plt.axes()
ax.errorbar(t, y, yerr=y_err, fmt='k.', label = 'Data')
T1 = np.linspace(t.min(), t.max(), 5000)
ax.plot(T1, linear(pf1, T1), 'r-', label = 'Fit')

ax.set_title('Nickel - $c_m/T$ vs $T^2$')
ax.set_xlabel('$T^2$')
ax.set_ylabel('$c_m/T$')
textfit = '$f(x) = A * T + B $ \n' \
          '$A = %.7f \pm %.7f $ \n' \
          '$B = %.0f \pm %.0f $ \n' \
          '$\chi^2 = % .2f$ \n' \
          '$N = % .2f$ \n' \
          '$\chi^2/N = % .2f$ \n' \
           % (pf1[0], pferr1[0], pf1[1], pferr1[1], chisq1, dof1, chisq1/dof1)
ax.text(0.05, .95, textfit, transform=ax.transAxes, fontsize=12,
         verticalalignment='top')
plt.legend(loc=4)         
plt.show()
plt.savefig('Ni_linfit.pdf')