import tensorflow as tf
import xlrd, xlsxwriter
import numpy as np
import time


def data_max_min():
    bg_max = np.array([1266.65154403, 1909.16496156, 2720.24454152, 3749.77812267, 4361.75138086, 5543.29628669]).reshape(1, 6)
    bg_min = np.array([12.05763547, 13.94927364, 24.44068356, 27.8992761, 36.92231046, 41.85357855]).reshape(1, 6)
    s_max = np.array([1.0e+03, 4.5e-01, 2.2e+00]).reshape(1, 3)
    s_min = np.array([1.02, 0.15, 1.5]).reshape(1, 3)
    ff_max = np.array([9.5e-01, 9.5e-01]).reshape(1, 2)
    ff_min = np.array([0.05, 0.05]).reshape(1, 2)
    return bg_max, bg_min, s_max, s_min, ff_max, ff_min


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


def save_designed_results(ff, a, s, t_bg, lv, filename):
    book = xlsxwriter.Workbook(filename)
    sheet1 = book.add_worksheet('designed_filling_fractions')
    sheet2 = book.add_worksheet('designed_periodic_constants')
    sheet3 = book.add_worksheet('soil_parameters')
    sheet4 = book.add_worksheet('targeted_bandgaps')
    sheet5 = book.add_worksheet('loss_value')
    for i in range(ff.shape[0]):
        for j in range(ff.shape[1]):
            sheet1.write(i, j, ff[i, j])
        for j in range(a.shape[1]):
            sheet2.write(i, j, a[i, j])
        for j in range(s.shape[1]):
            sheet3.write(i, j, s[i, j])
        for j in range(t_bg.shape[1]):
            sheet4.write(i, j, t_bg[i, j])
        for j in range(lv.shape[1]):
            sheet5.write(i, j, lv[i, j])
    book.close()


def inverse_network(t, alpha_ff, beta_ff, alpha_a, beta_a):
    with tf.variable_scope('inverse_design_network'):
        l = tf.layers.dense(t, 16)
        l = tf.layers.batch_normalization(l)
        l = tf.nn.tanh(l)

        # 用于输出填充比
        l_ff = tf.layers.dense(l, 1)
        l_ff = tf.layers.batch_normalization(l_ff)
        designed_ffc = (tf.sin(l_ff)+1)/2*(beta_ff-alpha_ff)+alpha_ff
        designed_ffs = 1-designed_ffc
        designed_ff = tf.concat([designed_ffc, designed_ffs], axis=1)

        # 用于输出周期常熟
        l_a = tf.layers.dense(l, 1)
        l_a = tf.layers.batch_normalization(l_a)
        designed_a = (tf.sin(l_a)+1)/2*(beta_a-alpha_a)+alpha_a

    return designed_ff, designed_a


def forward_modeling_network(p):
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


bg_max, bg_min, s_max, s_min, ff_max, ff_min = data_max_min()

x = tf.placeholder(tf.float32, [None, 2])  # 目标带隙投喂
c = tf.placeholder(tf.float32, [None, 3])  # 土体材料参数投喂

ff, a = inverse_network(x, 0.05, 0.95, 0.1, 40)  # ff是未归一化的填充比

ff_n = (ff-ff_min)/(ff_max-ff_min)
c_n = (c-s_min)/(s_max-s_min)

p = tf.concat([c_n, ff_n], axis=1)

pre_bg_n = forward_modeling_network(p)

pre_bg = pre_bg_n*(bg_max-bg_min)+bg_min

pre_bg = pre_bg/a

loss = tf.reduce_mean(tf.square(pre_bg[:, :2]-x))


#######  核心
t_vars = tf.trainable_variables()
updated_variables = [var for var in t_vars if var.name.startswith('inverse_design_network')]
optimizer = tf.train.AdamOptimizer(2e-4).minimize(loss, var_list=updated_variables)

restored_variables = [var for var in t_vars if var.name.startswith('pretrained_forward_modeling_network_for_P')]
saver = tf.train.Saver(restored_variables)
######  核心


with tf.Session() as sess:
    t_bgs, s_ps = load_targets('targets.xlsx')
    md = t_bgs.shape[0]

    designed_ffs, designed_as, lvs = np.zeros((md, 2)), np.zeros((md, 1)), np.zeros((md, 1))

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, 'pretrained_models/pretrained_forward_modeling_network_for_P_waves.ckpt')

    start = time.clock()
    for k in range(md):
        input_bg = t_bgs[k].reshape(1, 2)
        input_s = s_ps[k].reshape(1, 3)
        for epoch in range(1000):
            _, lv = sess.run([optimizer, loss], feed_dict={x: input_bg, c: input_s})
            if epoch > 100 and lv < 1e-2:
                break

        d_ff, d_a = sess.run([ff, a], feed_dict={x: input_bg})

        print(k, np.round(lv, 2))

        designed_ffs[k] = d_ff
        designed_as[k] = d_a
        lvs[k] = lv

    save_designed_results(designed_ffs, designed_as, s_ps, t_bgs, lvs, 'results/designed_parameters_for_P.xlsx')
    end = time.clock()
    print('It costs:', end-start, 's')
















