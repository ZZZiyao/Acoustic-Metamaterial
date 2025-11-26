import xlsxwriter
from sympy import *
import numpy as np


# 保存用于数据生成的参数变量集合
def save_parameters(x, filename):
    book = xlsxwriter.Workbook(filename)
    sheet = book.add_worksheet('sheet1')
    m, n = x.shape
    for i in range(m):
        for j in range(n):
            sheet.write(i, j, x[i, j])
    book.close()

n = 10000  # 设置生成参数的组数

# 生成土体和橡胶材料参数
Es = np.round(np.random.uniform(1, 1000, n), 2).reshape(-1, 1)  # MPa
Ps = np.round(np.random.uniform(0.15, 0.45, n), 3).reshape(-1, 1)
rhos = np.round(np.random.uniform(1.5, 2.2, n), 3).reshape(-1, 1)  # t/m^3
Er = np.round(np.random.uniform(0.1, 1, n), 2).reshape(-1, 1)  # MPa
Pr = np.round(np.random.uniform(0.45, 0.499, n), 3).reshape(-1, 1)
rhor = np.round(np.random.uniform(1.0, 1.4, n), 3).reshape(-1, 1)  # t/m^3

# 生成几何参数
L1, L2, L3, L4 = [], [], [], []  # 每层材料的填充吧
s1, s2, s3, s4 = 0.025, 0.05, 0.025, 0.05  # 每层材料的最小填充比

while True:
    l1 = np.round(np.random.uniform(s1,1-s2-s3-s4), 3)
    l2 = np.round(np.random.uniform(s2, 1-l1-s3-s4), 3)
    l3 = np.round(np.random.uniform(s3, 1-l1-l2-s4), 3)
    l4 = 1-l1-l2-l3
    L1.append(l1)
    L2.append(l2)
    L3.append(l3)
    L4.append(l4)
    if len(L1) == n:
        break

L1 = np.array(L1).reshape(-1, 1)
L2 = np.array(L2).reshape(-1, 1)
L3 = np.array(L3).reshape(-1, 1)
L4 = np.array(L4).reshape(-1, 1)


parameters = np.concatenate([Es, Ps, rhos, Er, Pr, rhor, L1, L2, L3, L4], axis=1)
print(parameters.shape)
save_parameters(parameters, 'parameters.xlsx')