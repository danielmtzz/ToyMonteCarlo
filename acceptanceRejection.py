import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


def pdf(x):
	return 3.0/8*(1+x**2)

xmax = minimize_scalar(lambda x: -pdf(x), bounds=[-1,1], method='bounded')


samplex = np.random.uniform(low=-1.0, high=1.0, size=(10000,))
sampley = np.random.uniform(low=0.0, high=xmax.x, size=(10000,))

test = sampley < pdf(samplex)
keep = samplex[np.where(test)[0]]
x = np.linspace(-1,1,5000)




fig, ax = plt.subplots()
ax = plt.subplot(111)
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
p1, = plt.plot(x,pdf(x),c='k',ls='--',label='PDF')
n , bins , patches = plt.hist(keep, 20, histtype='step', stacked=True, fill=False,density=True,
	linewidth=1.2,edgecolor='k')
patches[0].set_xy(patches[0].get_xy()[1:-1])
p2, = plt.plot([0,0], label='MC Simulation',linewidth=1.2,c='k')
#plt.scatter(dataCR113[:,1]*1000,pulseEnCR113,s=20,c='k')
leg = plt.legend([p1, p2], ['PDF', 'MC Simulation'],loc=(.6,.1),prop={'size': 13},fancybox=False, framealpha=1)
leg.get_frame().set_edgecolor('k')
p2.set_visible(False)
#bbox_props = dict(boxstyle="square",fc="white")
#ax.annotate('Wavemeter quoted accuracy = ' r'$\pm$' ' 2 MHz',
#	xy=(20,1.25), xytext=(20,1.25),size=13,bbox=bbox_props)
plt.title('Monte Carlo Acceptance-Rejection Simulation',size=16,fontname='Times New Roman')
plt.ylabel('Probability',size=14,fontname='Times New Roman')
plt.xlabel(r'$x$',size=14,fontname='Times New Roman')
plt.xticks(size=13,fontname='Times New Roman')
plt.yticks(size=13,fontname='Times New Roman')
ax.set_xlim(-1, 1) # apply the x-limits
#ax.set_ylim(-2, 2) # apply the y-limits
plt.minorticks_on()
ax.tick_params(which='both',direction='in',top=True,right=True)
plt.setp(ax.spines.values(), linewidth=1)
plt.tight_layout()
