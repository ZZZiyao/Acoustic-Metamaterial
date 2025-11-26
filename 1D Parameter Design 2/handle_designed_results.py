import numpy as np
import xlrd, xlsxwriter
import csv
import matplotlib.pylab as plt



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


def save_designed_bgs(d_bgs, filename):
    book = xlsxwriter.Workbook(filename)
    sheet1 = book.add_worksheet('designed_bandgaps')
    for i in range(d_bgs.shape[0]):
        for j in range(2):
            sheet1.write(i, j, d_bgs[i, j])
    book.close()


md = 20
d_bgs = np.zeros((md, 2))
for i in range(md):
    bgs = get_bandgaps('results/dispersion_curves_for_P/'+str(i+1)+'.csv')
    d_bgs[i] = bgs

save_designed_bgs(d_bgs, 'results/designed_bandgaps_for_P.xlsx')














