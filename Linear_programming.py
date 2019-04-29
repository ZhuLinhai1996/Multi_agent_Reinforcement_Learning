# encoding=utf-8

"""
Linear Programming
author: zhulinhai
使用scipy库中的方法进行线性规划方法优化
代码注意事项：
    c 待优化的函数式各变量系数
    A_ub 是<=不等式未知量的系数矩阵
    B_ub 是<=不等式的右边常量矩阵
    A_eq 是等式的未知量系数矩阵
    B_eq 是等式右边常量矩阵
    bounds 是每个未知量的范围
    注意：linprog()方法是求待优化函数的最小值，如果求最大值，则第一个传入参数c取负-c，则求出的就是c的最大值
"""
from scipy import optimize as op
import numpy as np
c=np.array([2,3,-5])
A_ub=np.array([[-2,5,-1],[1,3,1]])
B_ub=np.array([-10,12])
A_eq=np.array([[1,1,1]])
B_eq=np.array([7])
x1=(0,7)
x2=(0,7)
x3=(0,7)
res=op.linprog(-c,A_ub,B_ub,A_eq,B_eq,bounds=(x1,x2,x3))
print(res)


