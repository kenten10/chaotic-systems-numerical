from modules import one_dim_maps as maps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

def plot_graph_1_2(f, a):
    X = np.linspace(0.,1.,10**3)
    Y = np.array([f(a,x) for x in X])
    Y2 = np.array([f(a,y) for y in Y])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect(1.0 / ax.get_data_ratio())

    ax.plot(X, X, '-', color='k',label="$y=x$")
    ax.plot(X, Y, '-', color='r',label="$y=f(x)$")
    ax.plot(X, Y2, '-', color='b',label="$y=f^2(x)$")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.tick_params(labelsize=15)
    ax.set_xlabel("$x$", fontsize = 30)
    ax.set_ylabel("$y$", fontsize = 30)
    
    plt.tight_layout()
    # plt.show()
    plt.savefig("logistic_map_graphs_a%(a)1.5f.png"%{"a":a})
    plt.close()
def plot_graph_2_4(f, a):
    X = np.linspace(0.,1.,5*10**3)
    Y = np.array([f(a,x) for x in X])
    Y2 = np.array([f(a,y) for y in Y])
    Y3 = np.array([f(a,y2) for y2 in Y2])
    Y4 = np.array([f(a,y3) for y3 in Y3])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect(1.0 / ax.get_data_ratio())

    ax.plot(X, X, '-', color='k',label="$y=x$")
    ax.plot(X, Y2, '-', color='b',label="$y=f(x)$")
    ax.plot(X, Y4, '-', color='g',label="$y=f^2(x)$")
    
    vertices = []
    codes = []

    x0=0.36
    y0=0.36
    dist=0.15
    codes = [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]
    vertices = [(x0, y0), (x0, y0+dist), (x0+dist, y0+dist), (x0+dist, y0), (0, 0)]

    x0=0.815
    y0=0.815
    dist=0.075
    codes += [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]
    vertices += [(x0, y0), (x0, y0+dist), (x0+dist, y0+dist), (x0+dist, y0), (0, 0)]

    path = Path(vertices, codes)

    pathpatch = PathPatch(path, facecolor='none', edgecolor='black')

    ax.add_patch(pathpatch)
    
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.tick_params(labelsize=15)
    ax.set_xlabel("$x$", fontsize = 30)
    ax.set_ylabel("$y$", fontsize = 30)
    
    plt.tight_layout()
    # plt.show()
    plt.savefig("logistic_map_graphs_2_4_a%(a)1.5f.png"%{"a":a})
    plt.close()
def plot_graph_any(f, a):
    X = np.linspace(0.,1.,5*10**3)
    Y = np.array([f(a,x) for x in X])
    Y2 = np.array([f(a,y) for y in Y])
    Y3 = np.array([f(a,y2) for y2 in Y2])
    Y4 = np.array([f(a,y3) for y3 in Y3])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect(1.0 / ax.get_data_ratio())

    ax.plot(X, np.zeros(X.size), '-', color='k')
    ax.plot(X, Y - X, '-', color='r')
    ax.plot(X, Y2 - X, '-', color='b')
    ax.plot(X, Y4 - X, '-', color='g')
    ax.set_xlim(0,1)
    ax.set_ylim(-.5,.5)
    ax.tick_params(labelsize=15)
    ax.set_xlabel("$x$", fontsize = 30)
    ax.set_ylabel("$y$", fontsize = 30)
    
    plt.tight_layout()
    plt.show()
    # plt.savefig("logistic_map_graphs_2_4_a%(a)1.5f.png"%{"a":a})
    # plt.close()

# plot_graph_1_2(maps.logistic_map, 3.15)
plot_graph_2_4(maps.logistic_map, 3.5)
# plot_graph_any(maps.logistic_map, 3.45)

