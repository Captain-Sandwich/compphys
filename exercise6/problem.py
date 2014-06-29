#-*- coding: utf-8 -*-
"""
Created on Mon Jun 16 21:55:43 2014

@author: patrick
"""

import diffuse as df
from pylab import *
from util import set_plot_options
from mpl_toolkits.axes_grid.inset_locator import inset_axes
set_plot_options()

L = 1001
phi = np.zeros(L)
phi2 = np.zeros(L)
phi3 = np.zeros(L,dtype=np.int)
phi4 = np.zeros(L, dtype=np.int)
phi[(L+1)//2] = 1
phi2[0] = 1
phi3[(L+1)//2] = 10000
phi4[0] = 10000
tau = 0.001
m = 10000
D = 1
delta = .1
num_states = 1000
a, states, times = df.cdiffuse(phi, tau, m, delta, D, num_states=num_states)
b, states2, times2 = df.cdiffuse(phi2, tau, m, delta, D, num_states=num_states)
c = df.random_walkers(phi3, 100, delta)
d = df.random_walkers(phi4, 100, delta)

phi = np.zeros(L)
phi[(L+1)//2] = 1
g = df.cdiffuse(phi, tau, m, delta, 5)
g = g/delta**2

slopes = np.empty(50)
difs = linspace(0,30,50)
for i,d in enumerate(difs):
    phi = np.zeros(L)
    phi[(L+1)//2] = 1
    e = df.cdiffuse(phi, tau, 10, delta, d)/delta**2
    slopes[i] = polyfit(arange(1,11, dtype=np.double), e, 1)[0]

print slopes
slopes = slopes * delta**2 / (2*tau)
print slopes - difs

figure()
plot(difs, slopes)
vlines(1,0,1,alpha=0.5)
hlines(1,0,1,alpha=0.5)
# grid()
xlim(0,30)
ylim(0,10)
plt.table(cellText=[[r'$\tau$',tau], ['$\Delta$',delta]], colLabels=['Parameter','Value'], colWidths=[0.3,0.3], loc='lower right')
xlabel('Simulation Diffusion Constant')
ylabel('Fitted Parameter $D$')
savefig('latex/images/diffconstant.pdf')

#print "Erwartet für die random-walker:", 0.5*delta*delta/tau
# calc sigma'^2 = sigma^2 / delta^2 = 2 D t / delta^2 = 2 D t' tau / delta^2
ap = a/delta**2
bp = b/delta**2
cp = c#/delta**2
dp = d#/delta**2

t = arange(1,m+1)*tau
tp = arange(1,m+1, dtype=np.double)
x = (arange(L)-((L+1)//2))*delta
rwt = arange(1,101)*tau

#D =  [0.5*polyfit(t, i,1)[0] for i in [a,b]]
#D += [0.5*polyfit(rwt, i,1)[0] for i in [c,d]]
#D = np.array(D)
#print D

slopes2 = np.array([polyfit(tp, i, 1)[0] for i in [ap, bp]])
Ds = slopes2 * delta**2 / (2*tau)

print 'Slopes:', slopes2
print 'D: ', Ds

#slopes = np.array([polyfit(rwt, i, 1)[0] for i in [cp, dp]])

figure()
plot(t, ap, label='$\Phi_{L/2 + 1}=1$')
plot(t, bp, label='$\Phi_{1}=1$')
plot(t, polyval([200, 0], t), ls='--', color='k', label='analytical solution $D=1$')
ylabel(r'$\Delta^{-2} \left< x^2 \right> - \left< x \right>^2$')
# xlabel('$t^\prime$')
xlabel('t')
legend(loc='upper left')
# grid()
savefig('latex/images/productformula.pdf')


f = figure()
ax = f.add_subplot(111)
inset = f.add_axes([0.2, 0.4, 0.2, 0.3])
#inset = inset_axes(ax, width="25%", height="50%", loc='upper left')
idx = [0, 4, 9, 49, 99, 499, 999]
for i in idx:
    ax.plot(x, states[:,i], label='%.2f' % times[i])
    inset.plot(x, states[:,i], label='%.2f' % times[i])
ax.legend(loc='upper right')
ax.set_xlim(-5, 5)
ax.set_ylim(0, 0.14)
ax.set_xlabel('$x$')
ax.set_ylabel('$\Phi(x)$')
inset.set_xlim(-5, 5)
savefig('latex/images/cvstime.pdf')


f = figure()
ax = f.add_subplot(111)
inset = f.add_axes([0.5, 0.5, 0.2, 0.3])
#inset = inset_axes(ax, width="25%", height="50%", loc='upper left')
idx = [0, 4, 9, 49, 99, 499, 999]
for i in idx:
    ax.plot(x, states2[:,i], label='%.2f' % times2[i])
    inset.plot(x, states2[:,i], label='%.2f' % times2[i])
ax.legend(loc='lower right')
ax.set_xlim(-50.1, -48)
ax.set_ylim(0, 0.14)
ax.set_xlabel('$x$')
ax.set_ylabel('$\Phi(x)$')
inset.set_xlim(-50.1, -48)
savefig('latex/images/cvstime2.pdf')


figure()
plot(t, g, label='$\Phi_{L/2 + 1}=1$')
plot(rwt, cp, ls='-', lw=1, label='random walkers')
plot(t, polyval([1000, 0], t), ls='--', color='k', label='analytical solution $D=1$')
legend(loc='lower right')
# grid()
xlim(0, 100*tau)
ylim(0,100)
xlabel('Time $t$')
ylabel(r'$\Delta^{-2} \left< x^2 \right> - \left< x \right>^2$')
# savefig('latex/images/randomwalks.pdf')
show()
