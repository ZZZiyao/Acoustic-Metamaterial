import numpy as np
import xlrd, xlsxwriter
import csv
import matplotlib.pylab as plt


def R2(t, d):
    r2 = 1-np.sum(np.square(t-d))/np.sum(np.square(t-np.mean(t)))
    return r2


def RMSE(t, d):
    rmse = np.sqrt(np.mean(np.square(t-d)))
    return rmse

def get_bandgaps(filename):
    with open(filename, 'r') as f:
        data = csv.reader(f)
        rows = [row for row in data]
        rows = rows[8:]
        for i in range(len(rows)):
            rows[i] = float(rows[i][1])
        bg = np.zeros(2)
        bg[0], bg[1] = rows[1], rows[3]
    return bg


def load_targets(filename):
    book = xlrd.open_workbook(filename)
    sheet1 = book.sheet_by_name('targeted_bandgaps')
    sheet2 = book.sheet_by_name('soil_parameters')
    m, n1, n2 = sheet1.nrows, sheet1.ncols, sheet2.ncols
    t_bgs = np.zeros((m, n1))
    s = np.zeros((m, n2))
    for i in range(m):
        for j in range(n1):
            t_bgs[i, j] = sheet1.cell(i, j).value
        for j in range(n2):
            s[i, j] = sheet2.cell(i, j).value
    return t_bgs, s

def save_designed_bgs(d_bgs, t_bgs, e, filename):
    book = xlsxwriter.Workbook(filename)
    sheet1 = book.add_worksheet('designed_bandgaps')
    sheet2 = book.add_worksheet('targeted_bandgaps')
    sheet3 = book.add_worksheet('errors')
    for i in range(d_bgs.shape[0]):
        for j in range(2):
            sheet1.write(i, j, d_bgs[i, j])
            sheet2.write(i, j, t_bgs[i, j])
        sheet3.write(i, 0, e[i])
    book.close()


t_bgs, s = load_targets('targets.xlsx')
md = t_bgs.shape[0]
d_bgs = np.zeros((md, 2))
for i in range(md):
    bgs = get_bandgaps('results/dispersion_curves_for_P/'+str(i+1)+'.csv')
    d_bgs[i] = bgs

e = np.mean(np.abs(d_bgs-t_bgs)/t_bgs, axis=1)
save_designed_bgs(d_bgs, t_bgs, e, 'results/designed_bandgaps_for_P.xlsx')
r2 = R2(t_bgs, d_bgs)
rmse = RMSE(t_bgs, d_bgs)
print('R2:', r2, '\n', 'RMSE:', rmse, '\n', 'Error:', np.mean(e))














