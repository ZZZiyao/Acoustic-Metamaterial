import tensorflow as tf
import xlrd, xlsxwriter
import numpy as np
import time


# 归一化数据集中的数据
def normalized_data(x):
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    x = (x-x_min)/(x_max-x_min)
    return x, x_max, x_min



def normalized_dc(x):
    x_max = np.max(x)
    x_min = np.min(x)
    x = (x-x_min)/(x_max-x_min)
    return x, x_max, x_min


# 去归一化
def denormalized_data(x, x_max, x_min):
    return x*(x_max-x_min)+x_min


def load_data(num):
    book = xlrd.open_workbook('dataset/dataset_for_2D_DC.xlsx')
    sheet1 = book.sheet_by_name('images')
    sheet2 = book.sheet_by_name('soil_parameters')
    sheet3 = book.sheet_by_name('dispersion_curves')
    m, n1, n2, n3 = num, sheet1.ncols, sheet2.ncols, sheet3.ncols
    images = np.zeros((m, n1))
    sps = np.zeros((m, n2))
    dcs = np.zeros((m, n3))
    for i in range(m):
        for j in range(n1):
            images[i, j] = sheet1.cell(i, j).value
        for j in range(n2):
            sps[i, j] = sheet2.cell(i, j).value
        for j in range(n3):
            dcs[i, j] = sheet3.cell(i, j).value

    images = images.reshape(-1, 40, 40, 1)
    sps, sp_max, sp_min = normalized_data(sps)
    dcs, dc_max, dc_min = normalized_dc(dcs)

    tr_ims, va_ims, te_ims = images[:int(m * 0.8)], images[int(m * 0.8):int(m * 0.9)], images[int(m * 0.9):m]
    tr_sps, va_sps, te_sps = sps[:int(m * 0.8)], sps[int(m * 0.8):int(m * 0.9)], sps[int(m * 0.9):m]
    tr_dcs, va_dcs, te_dcs = dcs[:int(m * 0.8)], dcs[int(m * 0.8):int(m * 0.9)], dcs[int(m * 0.9):m]
    return tr_ims, va_ims, te_ims, tr_sps, va_sps, te_sps, tr_dcs, va_dcs, te_dcs, dc_max, dc_min, sp_max, sp_min


def calculate_accuracy(x1, x2):
    a = np.abs(x1-x2)
    b = x2
    acc = 1.0-np.mean(np.divide(a, b, out=np.zeros_like(a), where=b!=0), axis=1)
    return acc


def CNN(ims, sp):
    # 输出维度=（输入维度-kernel维度+2*padding）/strides+1
    x = tf.layers.conv2d(ims, filters=16, kernel_size=3, strides=2)  # 19
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)  # 9
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)

    x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=2)  # 4
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)  # 2
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)

    x = tf.layers.flatten(x)  # 输出维度 2*2*32=128
    x = tf.concat([x, sp], axis=1)

    x = tf.layers.dense(x, 256)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)

    x = tf.layers.dense(x, 186)
    x = tf.layers.batch_normalization(x)
    output = tf.nn.sigmoid(x)
    return output

def training_model(num):
    tr_ims, va_ims, te_ims, tr_sps, va_sps, te_sps, tr_dcs, va_dcs, te_dcs, dc_max, dc_min, sp_max, sp_min = load_data(num)
    x_ims = tf.placeholder(tf.float32, [None, 40, 40, 1])
    x_sps = tf.placeholder(tf.float32, [None, 3])
    y = tf.placeholder(tf.float32, [None, 186])

    output = CNN(x_ims, x_sps)

    loss = tf.reduce_mean(tf.square(output-y))
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)

    tr_acc, va_acc, tr_loss_history, va_loss_history = [], [], [], []

    saver = tf.train.Saver()

    with tf.Session() as sess:
        b_size = 500
        b_num = int(tr_ims.shape[0]/b_size)
        sess.run(tf.global_variables_initializer())
        start = time.clock()
        for epoch in range(1000):
            for k in range(b_num):
                sess.run(optimizer, feed_dict={x_ims: tr_ims[k*b_size:(k+1)*b_size], x_sps: tr_sps[k*b_size:(k+1)*b_size], y: tr_dcs[k*b_size:(k+1)*b_size]})

            if (epoch+1)%10 == 0:
                tr_loss_value = sess.run(loss, feed_dict={x_ims: tr_ims[:1000], x_sps: tr_sps[:1000], y: tr_dcs[:1000]})
                va_loss_value = sess.run(loss, feed_dict={x_ims: va_ims[:1000], x_sps: va_sps[:1000], y: va_dcs[:1000]})
                tr_loss_history.append(tr_loss_value)
                va_loss_history.append(va_loss_value)

                tr_p = sess.run(output, feed_dict={x_ims: tr_ims[:1000], x_sps: tr_sps[:1000]})
                va_p = sess.run(output, feed_dict={x_ims: va_ims[:1000], x_sps: va_sps[:1000]})
                tr_acc.append(np.mean(calculate_accuracy(denormalized_data(tr_p, dc_max, dc_min), denormalized_data(tr_dcs[:1000], dc_max, dc_min))))
                va_acc.append(np.mean(calculate_accuracy(denormalized_data(va_p, dc_max, dc_min), denormalized_data(va_dcs[:1000], dc_max, dc_min))))
                print(epoch+1, '  训练集精度|验证集精度：', np.round(tr_acc[-1]*100, 2), '%|', np.round(va_acc[-1]*100, 2), '%')

            if (epoch+1)%20 == 0:
                saver.save(sess, 'models/CNN.ckpt')
                print('CNN has been saved!')

                book = xlsxwriter.Workbook('results/learning_curves.xlsx')
                sheet = book.add_worksheet('learning_curves')
                for i in range(len(tr_loss_history)):
                    sheet.write(i, 0, tr_loss_history[i])
                    sheet.write(i, 1, va_loss_history[i])
                    sheet.write(i, 2, tr_acc[i])
                    sheet.write(i, 3, va_acc[i])
                book.close()
        end = time.clock()
        print('训练时间：', round(end-start, 4))


def predictor(num):
    tf.reset_default_graph()
    tr_ims, va_ims, te_ims, tr_sps, va_sps, te_sps, tr_dcs, va_dcs, te_dcs, dc_max, dc_min, sp_max, sp_min = load_data(num)
    x_ims = tf.placeholder(tf.float32, [None, 40, 40, 1])
    x_sps = tf.placeholder(tf.float32, [None, 3])
    output = CNN(x_ims, x_sps)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'models/CNN.ckpt')

        start = time.clock()
        predicted_dc = sess.run(output, feed_dict={x_ims: te_ims, x_sps: te_sps})
        end = time.clock()
        print('测试集预测用时：', np.round(end-start, 4))

    predicted_dc = denormalized_data(predicted_dc, dc_max, dc_min)
    te_dc = denormalized_data(te_dcs, dc_max, dc_min)

    book = xlsxwriter.Workbook('results/predicted_dispersion_curves.xlsx')
    sheet1 = book.add_worksheet('predictions')
    sheet2 = book.add_worksheet('labels')

    for i in range(te_dc.shape[0]):
        for j in range(te_dc.shape[1]):
            sheet1.write(i, j, predicted_dc[i, j])
            sheet2.write(i, j, te_dc[i, j])
    book.close()


# training_model(10000)
predictor(10000)