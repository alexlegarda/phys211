from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

def read_data(datafile):
    data = np.genfromtxt(datafile, delimiter=',', skiprows=1)
    return data
    
data = read_data(r'data/sample.csv')

t = data[:,0]
y = data[:,1]
y_err = data[:,2]

t1 = t[0:49]
y1 = y[0:49]
y_err1 = y_err[0:49]

t2 = t[50:98]
y2 = y[50:98]
y_err2 = y_err[50:98]

t3 = t[99:147]
y3 = y[99:147]
y_err3 = y_err[99:147]

def linear(p, x):
    return p[0] * (x - p[1])
def residual(p, x, y, err):
    return (linear(p, x) - y) / err
    
p1 = [0.1, 0.]
pf1, cov1, info1, mesg1, success1 = optimize.leastsq(residual, p1, args=(t1, y1, y_err1), full_output=1)
chisq1 = sum(info1["fvec"]*info1["fvec"])
dof1 = len(t1)-len(pf1)
pferr1 = [np.sqrt(cov1[i,i]) for i in range(len(pf1))]

p2 = [0.1, 0.]
pf2, cov2, info2, mesg2, success2 = optimize.leastsq(residual, p2, args=(t2, y2, y_err2), full_output=1)
chisq2 = sum(info2["fvec"]*info2["fvec"])
dof2 = len(t2)-len(pf2)
pferr2 = [np.sqrt(cov2[i,i]) for i in range(len(pf2))]

p3 = [0.1, 0.]
pf3, cov3, info3, mesg3, success3 = optimize.leastsq(residual, p3, args=(t3, y3, y_err3), full_output=1)
chisq3 = sum(info3["fvec"]*info3["fvec"])
dof3 = len(t3)-len(pf3)
pferr3 = [np.sqrt(cov3[i,i]) for i in range(len(pf3))]

print pf1, pf2, pf3



fig = plt.figure()
ax = fig.add_subplot(111)
ax = plt.axes()
ax.errorbar(t, y, yerr=y_err, fmt='k.', label = 'Data')
T1 = np.linspace(t1.min(), t1.max(), 5000)
T2 = np.linspace(t2.min(), t2.max(), 5000)
T3 = np.linspace(t3.min(), t3.max(), 5000)
ax.plot(T1, linear(pf1, T1), 'r-', label = 'Pre-Pulse')
ax.plot(T2, linear(pf2, T2), 'b-', label = '5s Pulse')
ax.plot(T3, linear(pf3, T3), 'g-', label = 'Post-Pulse')

ax.set_title('Nickel at 4.7 Kelvin')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Temperature (K)')
textfit = '$T_1 = A_1 * t + B_1 $ \n' \
          '$A_1 = %.3f \pm %.3f $ \n' \
          '$B_1 = %.1f \pm %.1f $ \n' \
          '$\chi^2_1/N_1 = % .2f$ \n' \
          '$T_2 = A_2 * t + B_2 $ \n' \
          '$A_2 = %.3f \pm %.3f $ \n' \
          '$B_2 = %.1f \pm %.1f $ \n' \
          '$\chi^2_2/N_2 = % .2f$ \n' \
          '$T_3 = A_3 * t + B_3 $ \n' \
          '$A_3 = %.3f \pm %.3f $ \n' \
          '$B_3 = %.1f \pm %.1f $ \n' \
          '$\chi^2_3/N_3 = % .2f$' \
           % (pf1[0], pferr1[0], pf1[1], pferr1[1], chisq1/dof1,
           pf2[0], pferr2[0], pf2[1], pferr2[1], chisq2/dof2,
           pf3[0], pferr3[0], pf3[1], pferr3[1], chisq3/dof3)
ax.text(0.05, .95, textfit, transform=ax.transAxes, fontsize=12,
         verticalalignment='top')
plt.legend(loc=4)         
plt.show()
plt.savefig('low-fit.pdf')