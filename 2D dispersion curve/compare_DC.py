import numpy as np
from matplotlib.pylab import plt
import xlrd


def load_data(filename):
    book = xlrd.open_workbook(filename)
    sheet1 = book.sheet_by_name('predictions')
    sheet2 = book.sheet_by_name('labels')
    m, n = sheet1.nrows, sheet2.ncols
    pre = np.zeros((m, n))
    label = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            pre[i, j] = sheet1.cell(i, j).value
            label[i, j] = sheet2.cell(i, j).value
    return pre, label

def plot_comparison(pre, label):
    x = np.arange(31)
    x = np.tile(x, 6)
    for i in range(pre.shape[0]):
        plt.scatter(x, label[i], s=20, c='black', label='Labels')
        plt.scatter(x, pre[i], s=10, c='red', label='Predictions')
        plt.xlabel('k')
        plt.ylabel('f (Hz)')
        plt.title('Comparison of True Curve and Predicted Points')
        plt.legend()
        plt.savefig('results/comparisons/'+str(i)+'.jpg')
        plt.close()

pre, label = load_data('results/predicted_dispersion_curves.xlsx')
pre, label = pre[:10], label[:10]
plot_comparison(pre, label)