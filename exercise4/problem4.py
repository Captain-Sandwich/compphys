# -*- coding: utf-8 -*-
"""
problem 4: coupled oscillators

@author: bethke
"""

import numpy as np
from integrators import mvverlet, multienergy
from pylab import *
from util import set_plot_options

set_plot_options()

def experiment1(N,T=10,dt=0.01):
    '''first starting condition: x(N/2,0) = 1'''
    t = np.arange(0, T+dt, dt)
    x = np.zeros((t.size, N))
    v = np.zeros((t.size, N))
    
    #initial conditions:
    x[0, N/2] = 1.

    # the integrators work in-place
    mvverlet(x, v, dt)
    E = multienergy(x,v)
    
    return t,x,v,E
    
def experiment2(N,T=10, dt=0.01, var='a'):
    '''second starting condition'''
    t = np.arange(0, T+dt, dt)
    x = np.zeros((t.size, N))
    v = np.zeros((t.size, N))
    
    #initial conditions:
    if var=='a':
        x[0,] = np.sin(np.pi * 1 * np.arange(1,N+1,dtype=np.double)/(N+1) )
    else:
        x[0,] = np.sin(np.pi * float(N/2) * np.arange(1,N+1,dtype=np.double)/(N+1) )
    
    mvverlet(x, v, dt)
    E = multienergy(x,v)
    
    return t,x,v,E    

######################################################################
# Everything from here on is just plotting and calling experiment1/2 #
######################################################################
#%% 128 Oscillators
T = 100
dt = 0.01
N = 128
t,x,v,E = experiment1(N,T=T,dt=dt)
del v,E

# Make a nice figure
f = figure()
f.set_size_inches(f.get_size_inches()*[2.1,2])
slices = 10
slice_times = linspace(0,T,slices,endpoint=False)
gs = mpl.gridspec.GridSpec(slices,3, width_ratios=[0.1,2,1])
gs.update(hspace=0.0, wspace=0.0)
ax1=subplot(gs[:,1])
im = ax1.imshow(x, interpolation='nearest', aspect='auto')
for t in slice_times:
    ax1.plot(np.ones(N)*t/dt, lw=0.8, color='white')
ax1.margins(0,0,tight=True)
xticks([])
yticks([])
axes = [subplot(gs[0,2])]
axes += [subplot(gs[i,2], sharex=axes[0], sharey=axes[0]) for i in range(1,slices)]
for ax, t in zip(axes,slice_times):
    ax.plot(x[t/dt,:], marker='.', lw=0.5)
    ax.set_yticks([])
    ax.set_xticks([])

axcmap = subplot(gs[:,0])
xticks([])
#axcmap.yaxis.tick_left()
tick_params(top='off', left='on', bottom='off', right='off')
colorbar(mappable=im, cax=axcmap, ticklocation='left')
savefig('latex/images/p4_1_%d.pdf' % N)

#%% different figures for fewer oscillators
T = 10
dt = 0.01
N = [4,16]
for n in N:
    t,x,v,E = experiment1(n, T=T, dt=dt)
    
    figure()
    lines = plot(t,E)
    lines += plot(t,x, label='x')    
    labels = ['$E$'] + ['$x_%d$' % i for i in xrange(1,n+1)]
    #lines = lines + [eline]
    #legend()
    if n != 16:
        legend(lines,labels, loc='upper right')
    xlabel('Time')
    ylabel('Displacement')
    savefig('latex/images/p4_1_%d.pdf' % n)
    

#%% Second boundary condition
# 128 Oscillators
T = 250
dt = 0.01
N = 128
t,x,v,E = experiment2(N,T=T,dt=dt)
del v,E

# Make a nice figure
f = figure()
f.set_size_inches(f.get_size_inches()*[2.1,2])
slices = 10
slice_times = linspace(0,T,slices,endpoint=False)
gs = mpl.gridspec.GridSpec(slices,3, width_ratios=[0.1,2,1])
gs.update(hspace=0.0, wspace=0.0)
ax1=subplot(gs[:,1])
im = ax1.imshow(x, interpolation='nearest', aspect='auto')
for t in slice_times:
    ax1.plot(np.ones(N)*t/dt, lw=0.8, color='white')
ax1.margins(0,0,tight=True)
xticks([])
yticks([])
axes = [subplot(gs[0,2])]
axes += [subplot(gs[i,2], sharex=axes[0], sharey=axes[0]) for i in range(1,slices)]
for ax, t in zip(axes,slice_times):
    ax.plot(x[t/dt,:], marker='.', lw=0.5)
    ax.set_yticks([])
    ax.set_xticks([])

axcmap = subplot(gs[:,0])
xticks([])
#axcmap.yaxis.tick_left()
tick_params(top='off', left='on', bottom='off', right='off')
colorbar(mappable=im, cax=axcmap, ticklocation='left')
savefig('latex/images/p4_2_%d.pdf' % N)



#%% different figures for fewer oscillators
T = 100
dt = 0.01
N = [4,16]
for n in N:
    t,x,v,E = experiment2(n, T=T, dt=dt)
    
    figure()
    lines = plot(t,E)
    lines += plot(t,x, label='x')    
    labels = ['$E$'] + ['$x_%d$' % i for i in xrange(1,n+1)]
    #lines = lines + [eline]
    #legend()
    if n != 16:
        legend(lines,labels, loc='upper right')
    xlabel('Time')
    ylabel('Displacement')
    savefig('latex/images/p4_2_%d.pdf' % n)

    
##%%
#f = figure()
#index = np.arange(N)
#vx = np.zeros(N)
#images = []
#figure()
#for i, frame in enumerate(linspace(0, t.size-1, 1000)):
#    lines = scatter(index,x[frame,:],color='red')
#    arrows = quiver(index, x[frame,:], vx, 0.1*v[frame,:])#,alpha=0.5)
#    images.append([lines, arrows])
##close()
#im_ani = animation.ArtistAnimation(f, images, interval=10, blit=True)

plt.show()