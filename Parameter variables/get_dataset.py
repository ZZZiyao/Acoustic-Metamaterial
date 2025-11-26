import xlsxwriter, xlrd
import numpy as np
import csv


def load_dirname(m):
    path = 'dispersion_curves'
    dir_list = []
    for i in range(m):
        dir_list.append(path+'/'+str(i+1)+'.csv')
    return dir_list


def get_parameters(filename, m):
    book = xlrd.open_workbook(filename)
    sheet = book.sheet_by_name('sheet1')
    n = sheet.ncols
    x = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            x[i, j] = sheet.cell(i, j).value
    return x


def get_bandgaps(filename):
    with open(filename, 'r') as f:
        data = csv.reader(f)
        rows = [row for row in data]
        rows = rows[8:]
        for i in range(len(rows)):
            rows[i] = float(rows[i][1])

        bg = np.zeros(6)
        bg[0], bg[1] = rows[1], rows[3]
        bg[2], bg[3] = rows[2], rows[4]
        bg[4], bg[5] = rows[5], rows[7]
    return bg


def save_data(bgs, params, filename):
    book = xlsxwriter.Workbook(filename)
    sheet1 = book.add_worksheet('bandgap')
    sheet2 = book.add_worksheet('parameters')
    m = bgs.shape[0]
    n1 = bgs.shape[1]
    n2 = params.shape[1]
    for i in range(m):
        for j in range(n1):
            sheet1.write(i, j, bgs[i, j])
        for j in range(n2):
            sheet2.write(i, j, params[i, j])
    book.close()


m = 100
dir_list = load_dirname(m)
params = get_parameters('parameters.xlsx', m)
bgs = np.zeros((m, 6))

for i in range(m):
    bg = get_bandgaps(dir_list[i])
    bgs[i, :] = np.array(bg)

save_data(bgs, params, 'dataset_1D.xlsx')
