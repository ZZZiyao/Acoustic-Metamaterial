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


# 去归一化
def denormalized_data(x, x_max, x_min):
    return x*(x_max-x_min)+x_min


def load_data(num):  # num表示在‘dataset/images’下加载num个excel文件，其中一个excel文件包含10000组拓扑数据。
    m = (num+1)*10000
    book1 = xlrd.open_workbook('dataset/dataset_for_2D_binary.xlsx')
    sheet1 = book1.sheet_by_name('in_plane')
    sheet2 = book1.sheet_by_name('soil_parameters')
    n1, n2 = 2, 3
    bgs = np.zeros((m, n1))
    sps = np.zeros((m, n2))
    for i in range(m):
        for j in range(n1):
            bgs[i, j] = sheet1.cell(i, j).value
        for j in range(n2):
            sps[i, j] = sheet2.cell(i, j).value
    images = np.zeros((m, 50*50))
    for k in range(num+1):
        book2 = xlrd.open_workbook('dataset/images/'+str(k)+'.xlsx')
        sheet3 = book2.sheet_by_name('images')
        for i in range(10000):
            for j in range(50*50):
                images[k*10000+i, j] = sheet3.cell(i, j).value
    bgs, bg_max, bg_min = normalized_data(bgs)
    sps, sp_max, sp_min = normalized_data(sps)
    images = images.reshape(-1, 50, 50, 1)
    tr_ims, va_ims, te_ims = images[:int(m*0.8)], images[int(m*0.8):int(m*0.9)], images[int(m*0.9):m]
    tr_sps, va_sps, te_sps = sps[:int(m*0.8)], sps[int(m*0.8):int(m*0.9)], sps[int(m*0.9):m]
    tr_bgs, va_bgs, te_bgs = bgs[:int(m*0.8)], bgs[int(m*0.8):int(m*0.9)], bgs[int(m*0.9):m]
    return tr_ims, va_ims, te_ims, tr_sps, va_sps, te_sps, tr_bgs, va_bgs, te_bgs, bg_max, bg_min, sp_max, sp_min


def calculate_accuracy(x1, x2):
    a = np.abs(x1-x2)
    b = x2
    acc = 1.0-np.mean(np.divide(a, b, out=np.zeros_like(a), where=b!=0), axis=1)
    return acc

# 构建卷积神经网络
def CNN(ims, sp):
    # 输出维度=（输入维度-kernel维度+2*padding）/strides+1
    x = tf.layers.conv2d(ims, filters=16, kernel_size=3, strides=2)  # 24
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)  # 12
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)

    x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=2)  # 5
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)  # 2
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)

    x = tf.layers.flatten(x)  # 输出维度 2*2*32=128
    x = tf.concat([x, sp], axis=1)

    # x = tf.layers.dense(x, 256)
    # x = tf.layers.batch_normalization(x)
    # x = tf.nn.relu(x)
    #
    # x = tf.layers.dense(x, 128)
    # x = tf.layers.batch_normalization(x)
    # x = tf.nn.relu(x)

    x = tf.layers.dense(x, 64)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)

    x = tf.layers.dense(x, 2)
    x = tf.layers.batch_normalization(x)
    output = tf.nn.sigmoid(x)
    return output


def training_model(num):
    tr_ims, va_ims, te_ims, tr_sps, va_sps, te_sps, tr_bgs, va_bgs, te_bgs, bg_max, bg_min, sp_max, sp_min = load_data(num)
    x_ims = tf.placeholder(tf.float32, [None, 50, 50, 1])
    x_sps = tf.placeholder(tf.float32, [None, 3])
    y = tf.placeholder(tf.float32, [None, 2])

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
                sess.run(optimizer, feed_dict={x_ims: tr_ims[k*b_size:(k+1)*b_size], x_sps: tr_sps[k*b_size:(k+1)*b_size], y: tr_bgs[k*b_size:(k+1)*b_size]})

            if (epoch+1)%10 == 0:
                tr_loss_value = sess.run(loss, feed_dict={x_ims: tr_ims[:1000], x_sps: tr_sps[:1000], y: tr_bgs[:1000]})
                va_loss_value = sess.run(loss, feed_dict={x_ims: va_ims, x_sps: va_sps, y: va_bgs})
                tr_loss_history.append(tr_loss_value)
                va_loss_history.append(va_loss_value)

                tr_p = sess.run(output, feed_dict={x_ims: tr_ims[:1000], x_sps: tr_sps[:1000]})
                va_p = sess.run(output, feed_dict={x_ims: va_ims, x_sps: va_sps})
                tr_acc.append(np.mean(calculate_accuracy(denormalized_data(tr_p, bg_max, bg_min), denormalized_data(tr_bgs[:1000], bg_max, bg_min))))
                va_acc.append(np.mean(calculate_accuracy(denormalized_data(va_p, bg_max, bg_min), denormalized_data(va_bgs, bg_max, bg_min))))
                print(epoch+1, '  训练集精度|验证集精度：', np.round(tr_acc[-1]*100, 2), '%|', np.round(va_acc[-1]*100, 2), '%')

            if (epoch+1)%20 == 0:
                saver.save(sess, 'models/CNN_for_in.ckpt')
                print('CNN has been saved!')

                book = xlsxwriter.Workbook('results/learning_curves_for_in_by_CNN.xlsx')
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
    tr_ims, va_ims, te_ims, tr_sps, va_sps, te_sps, tr_bgs, va_bgs, te_bgs, bg_max, bg_min, sp_max, sp_min = load_data(num)
    x_ims = tf.placeholder(tf.float32, [None, 50, 50, 1])
    x_sps = tf.placeholder(tf.float32, [None, 3])
    output = CNN(x_ims, x_sps)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'models/CNN_for_in.ckpt')

        start = time.clock()
        predicted_bg = sess.run(output, feed_dict={x_ims: te_ims, x_sps: te_sps})
        end = time.clock()
        print('测试集预测用时：', np.round(end-start, 4))

    predicted_bg = denormalized_data(predicted_bg, bg_max, bg_min)
    te_bg = denormalized_data(te_bgs, bg_max, bg_min)
    te_acc = calculate_accuracy(predicted_bg, te_bg)

    book = xlsxwriter.Workbook('results/prediction_accuracy_for_test_in_by_CNN.xlsx')
    sheet1 = book.add_worksheet('testing_set')

    for i in range(te_acc.shape[0]):
        sheet1.write(i, 0, te_acc[i])
        for j in range(predicted_bg.shape[1]):
            sheet1.write(i, 1+j, te_bg[i, j])
            sheet1.write(i, 1+predicted_bg.shape[1]+j, predicted_bg[i, j])
    book.close()

    print('测试集预测精度：', np.round(np.mean(te_acc)*100, 2), '%')


training_model(0)
predictor(0)
















