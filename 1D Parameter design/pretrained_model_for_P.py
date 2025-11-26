import tensorflow as tf
import xlrd
import numpy as np


# 归一化数据
def normalized_data(x):
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    x = (x-x_min)/(x_max-x_min)
    return x, x_max, x_min


# 数据去归一化
def denormalized_data(x, x_max, x_min):
    return x*(x_max-x_min)+x_min


# 加载数据
def load_data(filename, sheetname1, sheetname2, num):
    book = xlrd.open_workbook(filename)
    sheet1 = book.sheet_by_name(sheetname1)
    sheet2 = book.sheet_by_name(sheetname2)
    m, n1, n2 = num, 6, sheet2.ncols
    bandgaps = np.zeros((m, n1))  # 用于保存带隙数据
    parameters = np.zeros((m, n2))  # 用于保存材料和几何参数数据
    # 将excel表格的相关数据赋予到变量bandgaps和parameters中
    for i in range(m):
        for j in range(n1):
            bandgaps[i, j] = sheet1.cell(i, j).value
        for j in range(n2):
            parameters[i, j] = sheet2.cell(i, j).value
    # 将数据进行归一化
    bandgaps, bg_max, bg_min = normalized_data(bandgaps)
    parameters, param_max, param_min = normalized_data(parameters)

    # 为了进一步确保数据的随机性，对数据进行洗牌
    for i in range(n1):
        np.random.seed(1)
        np.random.shuffle(bandgaps[:, i])
    for i in range(n2):
        np.random.seed(1)
        np.random.shuffle(parameters[:, i])

    # 划分训练集、验证集、测试集
    tr_bg, va_bg, te_bg = bandgaps[:int(num*0.8)], bandgaps[int(num*0.8):int(num*0.9)], bandgaps[int(num*0.9):num]
    tr_param, va_param, te_param = parameters[:int(num*0.8)], parameters[int(num*0.8):int(num*0.9)], parameters[int(num*0.9):num]
    return tr_bg, va_bg, te_bg, tr_param, va_param, te_param, bg_max, bg_min, param_max, param_min


# 计算预测带隙的精度
def calculate_accuracy(x1, x2):
    a = np.abs(x1-x2)
    b = x2
    acc = 1.0-np.mean(np.divide(a, b, out=np.zeros_like(a), where=b!=0), axis=1)  # 如果b中的某个元素为零，则输出零，避免输出inf或None
    return acc


# 构建用于带隙预测的前向模拟网络
def MLP(p):
    # 非常重要！！！给前向模拟网络内部所有参数（权重w和偏差b）命名相同的“姓氏”，方便TNN运行时所有参数的调用
    with tf.variable_scope('pretrained_forward_modeling_network_for_P_waves'):
        l1 = tf.layers.dense(p, 1024)
        l1 = tf.layers.batch_normalization(l1)
        l1 = tf.nn.relu(l1)

        l2 = tf.layers.dense(l1, 512)
        l2 = tf.layers.batch_normalization(l2)
        l2 = tf.nn.relu(l2)

        l3 = tf.layers.dense(l2, 256)
        l3 = tf.layers.batch_normalization(l3)
        l3 = tf.nn.relu(l3)

        l4 = tf.layers.dense(l3, 256)
        l4 = tf.layers.batch_normalization(l4)
        l4 = tf.nn.relu(l4)

        l5 = tf.layers.dense(l4, 128)
        l5 = tf.layers.batch_normalization(l5)
        l5 = tf.nn.relu(l5)

        l6 = tf.layers.dense(l5, 6)
        l6 = tf.layers.batch_normalization(l6)
        output = tf.nn.sigmoid(l6)
    return output


def training_model(num, iterations):  # num是指加载多少组数据，iterations是指训练多少代
    # 加载训练集、验证集和测试集
    tr_bg, va_bg, te_bg, tr_param, va_param, te_param, bg_max, bg_min, param_max, param_min = load_data('dataset/dataset_for_1D_binary.xlsx', 'P', 'parameters', num)

    # 定义前向模拟网络输入和输出的占位符
    x = tf.placeholder(tf.float32, [None, tr_param.shape[1]])
    y = tf.placeholder(tf.float32, [None, tr_bg.shape[1]])

    # 建立前向模拟网络输入和输出的映射关系
    output = MLP(x)

    # 定义均方差损失函数
    loss = tf.reduce_mean(tf.square(output-y))

    # 定义优化器
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)

    # 定义空列表，用于记录训练集和验证集的历史精度和损失函数
    tr_acc, va_acc, tr_loss_history, va_loss_history = [], [], [], []

    # 用于保存模型
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(iterations):
            # 投喂输入和输出数据，训练前向模型网络
            sess.run(optimizer, feed_dict={x: tr_param, y: tr_bg})

            # 每迭代10次输出一次训练集和验证集的精度
            if (epoch+1)%10 == 0:
                # 计算损失函数
                tr_loss_value = sess.run(loss, feed_dict={x: tr_param, y: tr_bg})
                va_loss_value = sess.run(loss, feed_dict={x: va_param, y: va_bg})
                tr_loss_history.append(tr_loss_value)
                va_loss_history.append(va_loss_value)

                # 计算预测精度
                tr_p = sess.run(output, feed_dict={x: tr_param})
                va_p = sess.run(output, feed_dict={x: va_param})
                tr_acc_value = calculate_accuracy(denormalized_data(tr_p, bg_max, bg_min), denormalized_data(tr_bg, bg_max, bg_min))
                va_acc_value = calculate_accuracy(denormalized_data(va_p, bg_max, bg_min), denormalized_data(va_bg, bg_max, bg_min))
                tr_acc.append(np.mean(tr_acc_value))
                va_acc.append(np.mean(va_acc_value))
                print(epoch+1, '  训练集精度|验证集精度：', np.round(tr_acc[-1]*100, 2), '%|', np.round(va_acc[-1]*100, 2), '%')

            if (epoch+1)%100 == 0:
                saver.save(sess, 'pretrained_models/pretrained_forward_modeling_network_for_P.ckpt')
                print('Pretrained model has been saved！')


# training_model(40000, 10000)

tr_bg, va_bg, te_bg, tr_param, va_param, te_param, bg_max, bg_min, param_max, param_min = load_data('dataset/dataset_for_1D_binary.xlsx', 'P', 'parameters', 40000)

print(bg_max, bg_min, param_max, param_min)







