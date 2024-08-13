from modules import one_dim_maps as maps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
mpl.rcParams['text.usetex'] = True


def get_orbit(f, a, x, N):
    orbit = np.ones(N) * np.nan
    for i in range(N):
        orbit[i] = x
        x = f(a, x)
    return orbit

def plot_graph_repetition(f, a, orbit):
    
    X = np.linspace(0.,1.,10**3)
    Y = np.array([f(a,x) for x in X])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect(1.0 / ax.get_data_ratio())

    ax.plot(X, Y, '-', color='k')
    ax.plot(X, X, '-', color='k')
    for i in range(orbit.size-1):
        ax.plot([orbit[i],orbit[i]],[orbit[i],orbit[i+1]], '-', color = 'k')
        ax.plot([orbit[i],orbit[i+1]],[orbit[i+1],orbit[i+1]], '-', color = 'k')
        
    # plt.savefig()
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.tick_params(labelsize=15)
    ax.set_xlabel("$x_n$", fontsize = 30)
    ax.set_ylabel("$x_{n+1}$", fontsize = 30)
    
    plt.tight_layout()
    # plt.show()
    plt.savefig("logistic_map_graph_repetition_a%(a)1.5f_x0%(x0)1.6f.png"%{"a":a,"x0":orbit[0]})
    plt.close()

def plot_time_series(a, orbit):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(-1,orbit.size+1)
    ax.set_ylim(-.025,1.025)
    
    time = np.arange(orbit.size)
    ax.plot(time, orbit, '.-', color ='k')
    ax.tick_params(labelsize=15)
    ax.set_xlabel("$n$", fontsize = 30)
    ax.set_ylabel("$x_n$", fontsize = 30)
    
    plt.tight_layout()
    plt.savefig("logistic_map_orbit_time_series_a%(a)1.5f_x0%(x0)1.6f.png"%{"a":a,"x0":orbit[0]})
    plt.close()

def plot_graph(orbit):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect(1.0 / ax.get_data_ratio())
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.tick_params(labelsize=15)
    ax.set_xlabel("$x_n$", fontsize = 30)
    ax.set_ylabel("$x_{n+1}$", fontsize = 30)

    for i in range(N-1):
        ax.plot(orbit[i], orbit[i+1], '.', color='k')
        # ax.plot([orbit[i],orbit[i]],[orbit[i],orbit[i+1]], '-', color = 'k')
        # ax.plot([orbit[i],orbit[i+1]],[orbit[i+1],orbit[i+1]], '-', color = 'k')
            
    plt.tight_layout()
    # plt.show()
    plt.savefig("logistic_map_plot_xn1_vs_xn.png")


# x0 = np.random.random()*0.5+0.5
# print(x0)
x0 = 0.2006594632543654
x = copy.deepcopy(x0)
# A = np.arange(0.5, 4., 1e-4)
# A = np.arange(3.4,3.6,1e-3)
A=[3.]
N = 10**3

for a in A:
  orbit = get_orbit(maps.logistic_map, a, x0, N)
  np.save("./data/logistic_map_orbit_a%(a)1.5f_x0%(x0)1.6f"%{"a":a,"x0":x0}, orbit)
  orbit = np.load("./data/logistic_map_orbit_a%(a)1.5f_x0%(x0)1.6f.npy"%{"a":a,"x0":x0})[:100]
  plot_graph_repetition(maps.logistic_map, a, orbit)
  plot_time_series(a, orbit)
