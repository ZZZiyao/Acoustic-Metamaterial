import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
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


# 加载数据集
def load_data(filename, sheetname1, sheetname2, num):
    book = xlrd.open_workbook(filename)
    sheet1 = book.sheet_by_name(sheetname1)
    sheet2 = book.sheet_by_name(sheetname2)
    m, n1, n2 = num, 2, sheet2.ncols
    bandgaps = np.zeros((m, n1))  # 用于保存带隙数据
    parameters = np.zeros((m, n2))  # 用于保存材料和几何参数数据
    # 将excel表格的相关数据赋予到变量bandgaps和parameters中
    for i in range(m):
        for j in range(n1):
            bandgaps[i, j] = sheet1.cell(i, j).value
        for j in range(n2):
            parameters[i, j] = sheet2.cell(i, j).value

    # 将数据归一化
    bandgaps, bg_max, bg_min = normalized_data(bandgaps)
    parameters, param_max, param_min = normalized_data(parameters)

    # 对数据进行洗牌
    for i in range(n1):
        np.random.seed(1)
        np.random.shuffle(bandgaps[:, i])
    for i in range(n2):
        np.random.seed(1)
        np.random.shuffle(parameters[:, i])

    # 划分训练集、验证集、测试集
    tr_bg, va_bg, te_bg = bandgaps[:int(num*0.8)], bandgaps[int(num*0.8):int(num*0.9)], bandgaps[int(num*0.9):num]
    tr_param, va_param, te_param = parameters[:int(num*0.8)], parameters[int(num*0.8):int(num*0.9)], parameters[int(num*0.9):num]
    return tr_bg, va_bg, te_bg, tr_param, va_param, te_param, bg_max, bg_min, param_max, param_max


# 计算预测带隙精度
def calculate_accuracy(x1, x2):
    a = np.abs(x1-x2)
    b = x2
    acc = 1.0-np.mean(np.divide(a, b, out=np.zeros_like(a), where=b!=0), axis=1)
    return acc



# 构建支持向量机，核函数采用高斯径向基核函数 o=exp(-(i-c)^2/(2d^2))
def SVM(p):
    # 隐含层神经元个数
    h_n = 1024
    # tile平铺函数：输入p的维度为n*5，输出p的维度为n*5120
    p = tf.tile(p, [1, h_n])
    # 将tile的p的维度转换为n*1024*5，用于与1024个神经元进行计算
    p = tf.reshape(p, [-1, h_n, 5])

    # 定义核函数的均值
    c = tf.Variable(tf.truncated_normal([h_n, 5], stddev=0.1))
    # 定义核函数的标准差
    delta = tf.Variable(tf.truncated_normal([h_n], stddev=0.1))
    # 输入数据与核函数均值的欧拉距离
    dist = tf.reduce_sum(tf.square(p-c), axis=2)
    # 方差
    delta_2 = tf.square(delta)
    # 构建高斯径向基核函数
    rbf_out = tf.exp(-dist/2/delta_2)

    # 定义输出层的权重和偏差
    w = tf.Variable(tf.truncated_normal([h_n, 2], stddev=0.1))
    b = tf.Variable(tf.zeros([2]))
    output = tf.matmul(rbf_out, w)+b
    return output


def training_model(num, iterations):
    # 加载训练集、验证集、测试集
    tr_bg, va_bg, te_bg, tr_param, va_param, te_param, bg_max, bg_min, param_max, param_max = load_data('dataset/dataset_for_1D_binary.xlsx', 'P', 'parameters', num)

    # 定义输入和输出占位符
    x = tf.placeholder(tf.float32, [None, tr_param.shape[1]])
    y = tf.placeholder(tf.float32, [None, tr_bg.shape[1]])

    # 建立输入和输出的关系
    output = SVM(x)

    # 定义均方差损失函数
    loss = tf.reduce_mean(tf.square(output-y))
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)

    # 定义空列表，用于记录训练集和验证集的历史精度和历史损失函数
    tr_acc = []
    va_acc = []
    tr_loss_history = []
    va_loss_history = []

    # 用于保存模型
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        start = time.clock()
        for epoch in range(iterations):
            # 投喂输入与输出数据，并训练SVM
            sess.run(optimizer, feed_dict={x: tr_param, y: tr_bg})

            if (epoch+1)%10 == 0:
                # 计算损失函数
                tr_loss_value = sess.run(loss, feed_dict={x: tr_param, y: tr_bg})
                va_loss_value = sess.run(loss, feed_dict={x: va_param, y: va_bg})
                tr_loss_history.append(tr_loss_value)
                va_loss_history.append(va_loss_value)

                # 计算预测精度
                tr_p = sess.run(output, feed_dict={x: tr_param})
                va_p = sess.run(output, feed_dict={x: va_param})
                tr_acc.append(np.mean(calculate_accuracy(denormalized_data(tr_p, bg_max, bg_min), denormalized_data(tr_bg, bg_max, bg_min))))
                va_acc.append(np.mean(calculate_accuracy(denormalized_data(va_p, bg_max, bg_min), denormalized_data(va_bg, bg_max, bg_min))))
                print(epoch+1, '  训练集精度|验证集精度', np.round(tr_acc[-1]*100, 2), '%|', np.round(va_acc[-1]*100, 2), '%')

            if (epoch+1)%100 == 0:
                saver.save(sess, 'models/SVM_for_P.ckpt')
                print('SVM has been saved!')

                # 记录学习历史
                book = xlsxwriter.Workbook('results/learning_curves_for_P_by_SVM.xlsx')
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
    tf.reset_default_graph()  # 重置网络
    tr_bg, va_bg, te_bg, tr_param, va_param, te_param, bg_max, bg_min, param_max, param_max = load_data('dataset/dataset_for_1D_binary.xlsx', 'P', 'parameters', num)
    x = tf.placeholder(tf.float32, [None, te_param.shape[1]])
    output = SVM(x)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'models/SVM_for_P.ckpt')

        start = time.clock()
        predicted_bg = sess.run(output, feed_dict={x: te_param})
        end = time.clock()
        print('测试集预测用时：', np.round(end-start, 4))

    predicted_bg = denormalized_data(predicted_bg, bg_max, bg_min)
    te_bg = denormalized_data(te_bg, bg_max, bg_min)
    te_acc = calculate_accuracy(predicted_bg, te_bg)

    book = xlsxwriter.Workbook('results/prediction_accuracy_for_test_P_by_SVM.xlsx')
    sheet1 = book.add_worksheet('testing_set')
    for i in range(te_acc.shape[0]):
        sheet1.write(i, 0, te_acc[i])
        for j in range(predicted_bg.shape[1]):
            sheet1.write(i, 1+j, te_bg[i, j])
            sheet1.write(i, 1+predicted_bg.shape[1]+j, predicted_bg[i, j])
    book.close()

    print('测试集预测精度：', np.round(np.mean(te_acc)*100, 2), '%')


# training_model(10000, 10000)
predictor(10000)
















