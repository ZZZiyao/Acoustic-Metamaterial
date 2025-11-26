import tensorflow as tf
import xlrd, xlsxwriter
import numpy as np


def normalized_data(x):
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    x = (x-x_min)/(x_max-x_min)
    return x, x_max, x_min


# 数据去归一化
def denormalized_data(x, x_max, x_min):
    return x*(x_max-x_min)+x_min


def calculate_accuracy(x1, x2):
    a = np.abs(x1-x2)
    b = x2
    acc = 1.0-np.mean(np.divide(a, b, out=np.zeros_like(a), where=b!=0), axis=1)  # 如果b中的某个元素为零，则输出零，避免输出inf或None
    return acc


def load_data(filename):
    book = xlrd.open_workbook(filename)
    sheet1 = book.sheet_by_name('latent_vectors')
    sheet2 = book.sheet_by_name('filling_fractions')
    m, n1, n2 = sheet1.nrows, sheet1.ncols, sheet2.ncols
    lvs = np.zeros((m, n1))
    ffs = np.zeros((m, n2))
    for i in range(m):
        for j in range(n1):
            lvs[i, j] = sheet1.cell(i, j).value
        for j in range(n2):
            ffs[i, j] = sheet2.cell(i, j).value

    lvs, lvs_max, lvs_min = normalized_data(lvs)
    ffs, ffs_max, ffs_min = normalized_data(ffs)

    tr_lvs, va_lvs, te_vs = lvs[:int(m*0.8)], lvs[int(m*0.8):int(m*0.9)], lvs[int(m*0.9):m]
    tr_ffs, va_ffs, te_ffs = ffs[:int(m*0.8)], ffs[int(m*0.8):int(m*0.9)], ffs[int(m*0.9):m]

    return tr_lvs, va_lvs, te_vs,  tr_ffs, va_ffs, te_ffs, lvs_max, lvs_min, ffs_max, ffs_min


def MLP(p):
    # 非常重要！！！给前向模拟网络内部所有参数（权重w和偏差b）命名相同的“姓氏”，方便TNN运行时所有参数的调用
    with tf.variable_scope('MLP_for_filling_fraction'):
        l1 = tf.layers.dense(p, 128)
        l1 = tf.layers.batch_normalization(l1)
        l1 = tf.nn.relu(l1)

        l2 = tf.layers.dense(l1, 64)
        l2 = tf.layers.batch_normalization(l2)
        l2 = tf.nn.relu(l2)

        l3 = tf.layers.dense(l2, 32)
        l3 = tf.layers.batch_normalization(l3)
        l3 = tf.nn.relu(l3)

        l4 = tf.layers.dense(l3, 16)
        l4 = tf.layers.batch_normalization(l4)
        l4 = tf.nn.relu(l4)

        l5 = tf.layers.dense(l4, 1)
        l5 = tf.layers.batch_normalization(l5)
        output = tf.nn.sigmoid(l5)
    return output


def training_model():
    tr_lvs, va_lvs, te_vs, tr_ffs, va_ffs, te_ffs, lvs_max, lvs_min, ffs_max, ffs_min = load_data('dataset/filling_fraction_dataset.xlsx')
    print(lvs_max, lvs_min, ffs_max, ffs_min)
    x_lv = tf.placeholder(tf.float32, [None, tr_lvs.shape[1]])
    y = tf.placeholder(tf.float32, [None, 1])

    output = MLP(x_lv)

    loss = tf.reduce_mean(tf.square(output-y))

    optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)

    tr_acc, va_acc, tr_loss_history, va_loss_history = [], [], [], []

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1000):
            sess.run(optimizer, feed_dict={x_lv: tr_lvs,  y: tr_ffs})
            if (epoch+1)%10 == 0:
                tr_loss_value = sess.run(loss, feed_dict={x_lv: tr_lvs, y: tr_ffs})
                va_loss_value = sess.run(loss, feed_dict={x_lv: va_lvs, y: va_ffs})
                tr_loss_history.append(tr_loss_value)
                va_loss_history.append(va_loss_value)

                tr_p = sess.run(output, feed_dict={x_lv: tr_lvs})
                va_p = sess.run(output, feed_dict={x_lv: va_lvs})
                tr_acc.append(np.mean(calculate_accuracy(denormalized_data(tr_p, ffs_max, ffs_min), denormalized_data(tr_ffs, ffs_max, ffs_min))))
                va_acc.append(np.mean(calculate_accuracy(denormalized_data(va_p, ffs_max, ffs_min), denormalized_data(va_ffs, ffs_max, ffs_min))))
                print(epoch, ' 训练集精度|验证集精度：', tr_acc[-1], '|', va_acc[-1])

            if (epoch+1)%100 ==0:
                saver.save(sess, 'pretrained_models/MLP_for_filling_fraction.ckpt')
                print('Model has been saved!')


training_model()

















