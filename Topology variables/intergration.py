import xlsxwriter, xlrd
import numpy as np


def load_data(filename):
    book = xlrd.open_workbook(filename)
    sheet1 = book.sheet_by_name('images')
    x = np.zeros((50, 50))
    for i in range(50):
        for j in range(50):
            x[i, j] = sheet1.cell(i, j).value
    return x.reshape(1, 2500)

def generate_soil_parameters(n):
    Es = np.round(np.random.uniform(1, 100, n), 2).reshape(-1, 1)
    Ps = np.round(np.random.uniform(0.3, 0.45, n), 3).reshape(-1, 1)
    rhos = np.round(np.random.uniform(1.5, 2.2, n), 3).reshape(-1, 1)
    params = np.concatenate([Es, Ps, rhos], axis=1)
    return params

def save_data(x, params, num, filename):
    book = xlsxwriter.Workbook(filename)
    sheet1 = book.add_worksheet('images')
    sheet2 = book.add_worksheet('soil_parameters')
    for i in range(num):
        for j in range(50*50):
            sheet1.write(i, j, x[i, j])
        for j in range(params.shape[1]):
            sheet2.write(i, j, params[i, j])
    book.close()


num = 4  #一个excel包含num个数据
for m in range(2):
    x = np.zeros((num, 2500))
    for n in range(num):
        t = load_data('images/excel/'+str(m*num+n)+'.xlsx')
        x[n] = t[0]
    params = generate_soil_parameters(num)
    save_data(x, params, num, 'integrated_images/'+str(m)+'.xlsx')
