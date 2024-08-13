from modules import one_dim_maps as maps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
mpl.rcParams['text.usetex'] = True

def get_orbit(f, a, x0, N):
    orbit = np.ones(N) * np.nan
    x = copy.deepcopy(x0)
    for i in range(N):
        orbit[i] = x
        x = f(a, x)
    return orbit

def plot_bifurcation_diagram(amin, amax, x0, N_trans, N, da = 1e-4):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    A = np.arange(amin, amax, da)

    for a in A:
        a_s = np.ones(N) * a
        orbit = get_orbit(maps.logistic_map, a, x0, N_trans + N)
        ax.plot(a_s, orbit[N_trans:N_trans + N], ',', color='k')
    
    ax.tick_params(labelsize=15)
    ax.set_xlabel("$a$", fontsize = 30)
    ax.set_ylabel("$x$", fontsize = 30)
    ax.set_ylim(-0.05, 1.05) # ax.set_ylim(0.25, 0.75) # ax.set_ylim(0.3, 0.4)
    
    plt.tight_layout()
    plt.savefig("logistic_map_bifurcation_diagram_amin%(amin)1.5f_amax%(amax)1.5f_x0%(x0)1.6f_da%(da)1.10f.png"%{"amin":amin,"amax":amax,"x0":orbit[0],"da":da})
    plt.close()





