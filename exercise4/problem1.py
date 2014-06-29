# -*- coding: utf-8 -*-
"""
problem 1: Harmonic oscillator
$m\frac{d^2x}{dt^2} = -k x$
problem 2: Euler-Cromer
problem 3: Verlet

@author: bethke
"""

import numpy as np
import itertools
import integrators #Integrators are written in cython for speed (~100x)
from pylab import *

m = 1
k = 1

dt_list = np.array([0.1, 0.01, 0.001]) # time steps
T = 1000 #simulated time: 10000s

dt = 0.1
t = np.arange(0, T+dt, dt)
x = np.zeros(t.size)
v = np.zeros(t.size)
x[0] = 0
v[0] = 1

#%%Problem 1
times = []
energies = []
xs = []
vs = []
for dt in dt_list:
    t = np.arange(0, T+dt, dt)
    x = np.zeros(t.size)
    v = np.zeros(t.size)
    #initial conditions
    x[0] = 0
    v[0] = 1
    x,v = integrators.euler(x, v, dt)
    E = integrators.energy(x, v)
    xs.append(x)
    vs.append(v)
    energies.append(E)
    times.append(t)

##############################################################################
# The other problems are analog:                                             #
# Prepare the arrays, fill in initial conditions, call the cython integrator #
##############################################################################

# This is just for plotting
rc('legend', fancybox=True, handlelength=1.2, borderaxespad=0.25, borderpad=0.2, columnspacing=1.0)
rc('font', size=11, family='Palatino Linotype')
rc('axes', color_cycle=['e41a1c', '377eb8', '4daf4a', '984ea3', 'ff7f00', 'ffff33', 'a65628'])
rc('figure', figsize=(2.83509486, 1.75208862348)) # one column size
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 7
rcParams['ytick.labelsize'] = 7
rcParams['legend.fontsize'] = 7
rc('axes', linewidth=0.4)
rc('xtick', direction='out')
rc('xtick.major', width=0.4, size=4)
rc('ytick', direction='out')
rc('ytick.major', width=0.4, size=4)
rc('grid', alpha=0.5, linestyle='-', linewidth=0.2)
rc('lines', markeredgewidth=0.2, markersize=2, linewidth=0.5)


#%% Plot time domain
figure()
#title('Energy Drift')
ax = subplot(111)
ax.set_position([0.2,0.25,0.75,0.7])
plot(times[0], xs[0], ls='..', marker='+', label='$x_{\\tau=%g}$' % dt_list[0])
plot(times[1], xs[1], ls='--', marker='', label='$x_{\\tau=%g}$' % dt_list[1])
plot(times[0], vs[0], ls='..', marker='x', label='$v_{\\tau=%g}$' % dt_list[0])
plot(times[1], vs[1], ls='--', marker='', label='$v_{\\tau=%g}$' % dt_list[1])
plot(t, sin(t), label='exact $x$', color='#222222', alpha=0.7)
lg = legend(loc='lower right')
fr = lg.get_frame()
fr.set_lw(0.2)
xlabel('Time')
ylabel('Position/Velocity')
xlim(0,10)
ylim(-1.5,1.5)
grid()
tick_params(top='off', right='off')
savefig('latex/images/time_domain_euler.pdf')
show()

#%% Plot phase space
figure()
gcf().set_figwidth(gcf().get_figheight())
#title('Energy Drift')
ax = subplot(111)
ax.set_position([0.25,0.25,0.7,0.7])
for i, dt in enumerate(dt_list[:1]):
    plot(xs[i], vs[i], label='$\\tau=%g$' % dt)
plot(sin(linspace(0,2*pi,100)), cos(linspace(0,2*pi,100)), label='exact', color='#222222')
lg = legend(loc='lower right')
fr = lg.get_frame()
fr.set_lw(0.2)
#axis('equal')
xlim(-2,2)
ylim(-2,2)
ax.set_yticks([-2,-1,0,1,2])
ax.set_xticks([-2,-1,0,1,2])
xlabel('Position')
ylabel('Velocity')
grid()
tick_params(top='off', right='off')
savefig('latex/images/phase_space_euler.pdf')
show()


#%% Plot Energy Drift
figure()
#title('Energy Drift')
ax = subplot(111)
ax.set_position([0.2,0.25,0.75,0.7])
for i, dt in enumerate(dt_list):
    plot(times[i], energies[i], label='$%g$' % dt)
xlim(0,10)
ylim(0,1)
lg = legend(loc='lower right')
fr = lg.get_frame()
fr.set_lw(0.2)
xlabel('Simulation time')
ylabel('Energy')
grid()
tick_params(top='off', right='off')
savefig('latex/images/E_vs_time.pdf')
show()

#figure()
#title('Energy Drift')
#for i, dt in enumerate(dt_list):
#    plot(energies[i], label='Timestep %.3f' % dt)
#xlim(0,1000)
#ylim(0,1)
#legend()
#xlabel('Simulation steps')
#ylabel('Energy')
#savefig('latex/images/E_vs_steps.pdf')
#show()

#%%Problem 2
T = 10
dt_list=[0.1, 0.01]
for dt,integrator in itertools.product(reversed(dt_list),[integrators.euler_cromer_a, integrators.euler_cromer_b]):
    t = np.arange(0, T+dt, dt)
    x = np.zeros(t.size)
    v = np.zeros(t.size)
    x[0] = 0
    v[0] = 1
    x,v = integrator(x, v, dt)

##%% Plots
figure()
ax = subplot(111)
ax.set_position([0.2,0.25,0.75,0.7])
plot(t, x, label='$x$', marker='x', ls='')
plot(t, v, label='$v$', marker='+', ls='')
plot(t, integrators.energy(x,v), label='$E$', ls='', marker='.', mfc='none', mec='#4daf4a')
plot(t, sin(t), alpha=0.7, lw=0.5, ls='-', color='#222222', label='exact $x$')
plot(t, np.zeros_like(t)+0.5, alpha=1, ls='-.', lw=1, color='#222222', label='exact $E$')
lg = legend(loc='lower right')
fr = lg.get_frame()
fr.set_lw(0.2)
xlim(0,10)
ylim(-1, 1)
xlabel('Time')
ylabel('$x(t)$, $v(t)$, $E(t)$')
tick_params(top='off', right='off')
grid()
#title('Euler-Cromer Algorithm')
#tight_layout()
savefig('latex/images/euler_cromer.pdf')
#show()
#
#%%Problem 3
dt_list = [0.1, 0.01]
for dt in reversed(dt_list):
    t = np.arange(0, T+dt, dt)
    x = np.zeros(t.size)
    v = np.zeros(t.size)
    x[0] = 0
    v[0] = 1
    x,v = integrators.verlet(x, v, dt)
#    np.savez_compressed('data/p3_%.3f'%dt, x=x, v=v, t=t, dt=dt)
#%% Verlet figure
figure()
ax = subplot(111)
ax.set_position([0.2,0.25,0.75,0.7])
plot(t, x, label='$x$', marker='x', ls='')
plot(t, v, label='$v$', marker='+', ls='')
plot(t, integrators.energy(x,v), label='$E$', ls='', marker='.', mfc='none', mec='#4daf4a')
plot(t, sin(t), alpha=0.7, lw=0.5, ls='-', color='#222222', label='exact $x$')
plot(t, np.zeros_like(t)+0.5, alpha=1, ls='-.', lw=1, color='#222222', label='exact $E$')
lg = legend(loc='lower right')
fr = lg.get_frame()
fr.set_lw(0.2)
xlim(0,10)
ylim(-1, 1)
xlabel('Time')
ylabel('$x(t)$, $v(t)$, $E(t)$')
a = gca()
#a.spines['right'].set_color('none')
#a.spines['top'].set_color('none')
a.yaxis.set_ticks_position('left')
a.xaxis.set_ticks_position('bottom')
grid()
savefig('latex/images/verlet.pdf')
#show()
#
#%% compare verlet and euler_cromer
T = 1000
dt = 0.01
t = np.arange(0, T+dt, dt)
x = np.zeros(t.size)
v = np.zeros(t.size)
x[0] = 0
v[0] = 1
x_euler_a, v_euler_a = integrators.euler_cromer_a(x, v, dt)
E_euler_a = integrators.energy(x_euler_a, v_euler_a)

x = np.zeros(t.size)
v = np.zeros(t.size)
x[0] = 0
v[0] = 1
x_euler_b, v_euler_b = integrators.euler_cromer_b(x, v, dt)
E_euler_b = integrators.energy(x_euler_b, v_euler_b)

x = np.zeros(t.size)
v = np.zeros(t.size)
x[0] = 0
v[0] = 1
x_euler, v_euler = integrators.euler(x, v, dt)
E_euler = integrators.energy(x_euler, v_euler)

x = np.zeros(t.size)
v = np.zeros(t.size)
x[0] = 0
v[0] = 1
x_verlet, v_verlet = integrators.velocity_verlet(x, v, dt)
E_verlet = integrators.energy(x_verlet, v_verlet)

# Plotting
figure()
ax = subplot(111)
ax.set_position([0.2,0.25,0.75,0.7])
plot(t, E_verlet, ls = ':', lw=1,  label='Verlet')
plot(t, E_euler_a, label='Euler-Cromer a)')
plot(t, E_euler_b, label='Euler-Cromer b)')
plot(t, E_euler, label='simple Euler')
plot(t, np.zeros_like(t)+0.5, color='#222222', ls='-')#, label='analytical $E$')
xlabel('Time')
ylabel('Energy')
ylim(0.49, 0.51)
xlim(0,10)
#title('Energy Comparison')
lg = legend(loc='lower right',ncol=2)
fr = lg.get_frame()
fr.set_lw(0.2)
ax.set_yticks([0.49, 0.5, 0.51])
tick_params(top='off', right='off')
savefig('latex/images/energy_verlet_euler.pdf')

figure()
ax = subplot(211)
plot(t, v_verlet, ls = ':', lw=1,  label='Verlet')
plot(t, v_euler_a, ls= '--', label='EC a)')
plot(t, cos(t), label='$\\sin(t)$', ls='-', alpha=0.5, color='#222222', zorder=-5)
ax.set_xlim(990,1000)
lg = ax.legend(loc='lower right', ncol=3)
fr = lg.get_frame()
fr.set_lw(0.2)
xlim(990,1000)
ylim(-1.5,1)
tick_params(top='off', right='off', bottom='off')
ax.set_xticks([])
ax.set_yticks([-1, 0, 1])
ylabel('$v$')

ax2 = subplot(212)
ax2.plot(t, (v_verlet-v_euler_a), label='Difference')
lg = ax2.legend(loc='lower right',ncol=2)
ax2.set_xlim(990,1000)
fr = lg.get_frame()
fr.set_lw(0.2)
tick_params(top='off', right='off')
ax2.set_yticks([-0.005, 0., 0.005])
ax2.set_yticklabels(['$-5 \\cdot 10^{-3}$', '$0$', '$5 \\cdot 10^{-3}$'])
ylabel('$\\Delta v$')
xlabel('Time')
tight_layout()
gcf().subplots_adjust(hspace=0.)
savefig('latex/images/velocity_verlet_euler.pdf')