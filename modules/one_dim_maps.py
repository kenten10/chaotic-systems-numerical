# Maps defined on one-dimensional Euclidean space.

import numpy as np

# Logistic map: x' = ax(1-x)
def logistic_map(a:float,x:float) -> float:
	return a * x * (1. - x)

# Logistic map (in: array, out: array)
def Logistic_Map(a:float, X:np.array) -> np.array:
	return a * X * (1. - X)


def doubling_map(x:float) -> float:
	xx = 2. * x
	return xx - np.floor(xx)

# doubling map: x' = 2x mod 1
def Doubling_Map(X:np.array) -> np.array:
	a = (X < 0.5) & (X >= 0.)

	b = (X < 1) & (X >= 0.5)
	
	# 第1項は区間[0,0.5]に含まれる要素に対しての計算で、
	# 第2項は区間[0.5,1]に含まれる要素に対しての計算
	XX = (2. * X * a) + (2. * (X - 1.) * b)
	
	return XX - np.floor(XX)

def tent_map(x:float) -> float:
	if (x < 0.) | (x > 1.):
		print("引数の範囲が不正です。")
	if (x < 0.5) & (x >= 0.):
		xx = 2. * x	
	elif (x < 1.) & (x >= 0.5):
		xx = -2. * x + 2.

	return xx - np.floor(xx)

def Tent_Map(x:np.array) -> np.array:
	a = (x < 0.5) & (x >= 0)
	b = (x < 1) & (x >= 0.5)
	xx = 2 * x * a + (-2 * x + 2) * b
	return xx

