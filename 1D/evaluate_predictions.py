import numpy as np
import matplotlib.pyplot as plt
import xlrd
from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'

def load_data(filename, sheetname):
    book = xlrd.open_workbook(filename)
    sheet = book.sheet_by_name(sheetname)
    m = sheet.nrows
    pre = np.zeros(m*2)
    label = np.zeros(m*2)
    for i in range(m):
       label[i*2] = sheet.cell(i, 1).value
       label[i*2+1] = sheet.cell(i, 2).value
       pre[i*2] = sheet.cell(i, 3).value
       pre[i*2+1] = sheet.cell(i, 4).value
    return label, pre


# r2 = 1-sum[(T-P)**2]/sum[(T-T_mean)**2]
def R2(P, T):
    # 计算总平方和 TSS
    tss = np.sum((T-np.mean(T))**2)
    # 计算残差平方和 RSS
    rss = np.sum((T-P)**2)
    # 计算决定系数 R2
    r2 = 1-(rss/tss)
    return r2

def plot(P, T):
    # 创建散点图
    plt.scatter(P, T, color='blue', label='预测值')
    # 绘制真实值的直线
    plt.plot(T, T, color='red', label='真实值')

    # 添加横坐标和纵坐标
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    r2 = R2(P, T)

    # 添加标签和标题
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.title(f'真实值 VS 预测值\n决定系数 $R^2$: {r2:.4f}')
    plt.show()

label, pre = load_data('results/prediction_accuracy_for_test_P_by_SVM.xlsx', 'testing_set')
plot(pre, label)



