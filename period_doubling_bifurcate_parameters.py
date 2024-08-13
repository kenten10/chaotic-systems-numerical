from modules import one_dim_maps as maps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

# f^N(x)のグラフを取得する
def get_graph(a, N, X0, F=maps.Logistic_Map):
    X = np.copy(X0)
    for _ in range(N):
        X = F(a, X)
    return X

# Yが0になるx座標の値を取得する
def get_zeros(X, Y):
    index = np.where(Y[:-1] * Y[1:] < 0.)
    a = (Y[:-1][index] - Y[1:][index])/(X[:-1][index] - X[1:][index])
    b = Y[:-1][index] - a * X[:-1][index]
    return -b / a

# f^Nの点x0における微係数を取得する
def get_differential_coefficient(a, x0, N, f=maps.logistic_map, dx=1e-6):
    x1 = x0 - .5 * dx
    x2 = x1 + dx
    for _ in range(N):
        x1 = f(a, x1)
        x2 = f(a, x2)
    return (x2 - x1) / dx

# 周期倍分岐が発生するパラメータaを取得する
def get_period_doubling_bifurcation_params(M, P, amin, amax = 4., da=1e-6):
    X0 = np.linspace(0., 1., M, endpoint=False)
    A = np.arange(amin, amax, da)

    # 周期倍分岐が発生するパラメータaを格納するリスト
    A_bifurcate = []
    for p in range(0,P):
        # print(p)
        for j, a in enumerate(A):
            # f^{2^p}のグラフを取得
            Y = get_graph(a, 2**p, X0)

            # f^{2^p} - xが0となる点=f^{2^p}の不動点を取得する
            zeros = get_zeros(X0, Y - X0)
            
            # f^{2^p}の不動点における微係数の絶対値を取得する
            diff_coeffs = np.array([np.abs(get_differential_coefficient(a, x, 2**p)) for x in zeros])
            # print(a)
            if j == 0:
                # 微係数の絶対値が1より小さい(吸引的)不動点の番号を取得する
                index = np.where(diff_coeffs < 1.)
            elif (diff_coeffs[index].size > 0) and (diff_coeffs[index] > 1.).all():
                # 周期倍分岐が発生=吸引的不動点が反発的不動点に変わる
                # 周期倍分岐が発生した直後のパラメータの値をA_bifurcateに格納する
                A_bifurcate.append(a)

                # 余計な計算を省くためにAを取り直す
                A = np.arange(a, amax, da)

                break

        print(A_bifurcate)
    return A_bifurcate


# A_bifurcate = get_period_doubling_bifurcation_params(10**5, 6, 2.999)
# print(A_bifurcate)

# 周期倍分岐が発生するパラメータ列{a_i}におけるd = (a_{i+1} - a_{i}) / (a_{i+2} - a_{i+1})を各iについて計算する
A_bifurcate = [3.00000100000014, 3.4494900000629687, 3.544091000076192, 3.5644080000790317, 3.56876000007964, 3.5696930000797704]
for i in range(len(A_bifurcate) - 2):
    d = (A_bifurcate[i+1] - A_bifurcate[i])/(A_bifurcate[i+2] - A_bifurcate[i+1])
    print(i, d)

def plot_A_bifurcate():
    Y = np.array([3.00000100000014, 3.4494900000629687, 3.544091000076192, 3.5644080000790317, 3.56876000007964, 3.5696930000797704])
    X = np.array([1.,2.,3.,4.,5.,6.])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_aspect(1.0 / ax.get_data_ratio())
    c0=3.56994567
    c1=2.628
    c2=1.543
    ax.plot(X, Y, 'o', color='k',label="numerical result")
    ax.plot(X, c0 - c1*np.exp(-c2*X),'x',markersize=10,color='r',label="$c_0-c_1e^{-c_2p}$")

    # ax.set_xlim(0,1)
    # ax.set_ylim(0,1)
    ax.tick_params(labelsize=15)
    ax.set_xlabel("$p$", fontsize = 30)
    ax.set_ylabel("$a_p$", fontsize = 30)
    ax.legend(fontsize=15)
    plt.tight_layout()
    # plt.show()
    plt.savefig("period_doubling_bifurcate_parameters.png")
    plt.close()

# plot_A_bifurcate()