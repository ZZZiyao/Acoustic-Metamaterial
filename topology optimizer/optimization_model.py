import tensorflow as tf
import xlsxwriter
from VAE import vae
from PIL import Image
import numpy as np


def data_max_min():
    bg_max = np.array([135.5399499, 302.14354031]).reshape(1, 2)
    bg_min = np.array([10.49087411, 12.22594684]).reshape(1, 2)
    s_max = np.array([99.9991248, 0.39999765, 2.19992069]).reshape(1, 3)
    s_min = np.array([1.00068476, 0.30000125, 1.60000726]).reshape(1, 3)
    lv_max = np.array([1.99965772, 1.99999302, 1.99985184]).reshape(1, 3)
    lv_min = np.array([-1.9996749, -1.99925434, -1.99986302]).reshape(1, 3)
    ff_max = 0.63679999
    ff_min = 0.2128
    return bg_max, bg_min, s_max, s_min, lv_max, lv_min, ff_max, ff_min


def save_designed_results(images, a, filename):
    images = images.reshape(-1, 50*50)
    book = xlsxwriter.Workbook(filename)
    sheet1 = book.add_worksheet('optimized_topology')
    sheet2 = book.add_worksheet('optimized_periodic_constants')
    for i in range(images.shape[0]):
        for j in range(images.shape[1]):
            sheet1.write(i, j, images[i, j])
        for j in range(a.shape[1]):
            sheet2.write(i, j, a[i, j])
    book.close()
    for i in range(images.shape[0]):
        im = images[i].reshape(50, 50)
        im = np.matrix(np.round(im), dtype='float')*255
        im = Image.fromarray(im.astype(np.uint8))
        im.save('results/optimized_topologies/'+str(i)+'.jpg')


def VAE_decoder(latent_vectors):
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 50, 50, 1])
    learning_rate = tf.placeholder(tf.float32)
    optimizer, loss, output, z_mean, z_log_var, z = vae(x, learning_rate)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'VAE/VAE.ckpt')
        images = sess.run(output, feed_dict={z: latent_vectors})
        images = np.round(images)
    return images


def inverse_network(alpha_a, beta_a):
    with tf.variable_scope('inverse_design_network'):

        t1 = tf.Variable(tf.constant(np.array([0.0, 0.0, 0.0]).reshape(1, 3), dtype=tf.float32), dtype=tf.float32)
        t2 = tf.Variable(tf.constant(np.array([0.0]).reshape(1, 1), dtype=tf.float32), dtype=tf.float32)

        # l = tf.layers.dense(t, 16)
        # l = tf.layers.batch_normalization(l)
        # l = tf.nn.tanh(l)

        # 用于输出潜向量
        l_lvs = tf.layers.dense(t1, 3)
        l_lvs = tf.layers.batch_normalization(l_lvs)
        designed_lvs_n = (tf.sin(l_lvs)+1)/2

        # 用于输出周期常熟
        l_a = tf.layers.dense(t2, 1)
        l_a = tf.layers.batch_normalization(l_a)
        designed_a = (tf.sin(l_a)+1)/2*(beta_a-alpha_a)+alpha_a
    return designed_lvs_n, designed_a


def BPM(p):
    # 非常重要！！！给前向模拟网络内部所有参数（权重w和偏差b）命名相同的“姓氏”，方便TNN运行时所有参数的调用
    with tf.variable_scope('pretrained_forward_modeling_network_for_out_of_plane_waves'):
        l1 = tf.layers.dense(p, 512)
        l1 = tf.layers.batch_normalization(l1)
        l1 = tf.nn.relu(l1)

        l2 = tf.layers.dense(l1, 256)
        l2 = tf.layers.batch_normalization(l2)
        l2 = tf.nn.relu(l2)

        l3 = tf.layers.dense(l2, 128)
        l3 = tf.layers.batch_normalization(l3)
        l3 = tf.nn.relu(l3)

        l4 = tf.layers.dense(l3, 64)
        l4 = tf.layers.batch_normalization(l4)
        l4 = tf.nn.relu(l4)

        l5 = tf.layers.dense(l4, 2)
        l5 = tf.layers.batch_normalization(l5)
        output = tf.nn.sigmoid(l5)
    return output


def FFPM(p):
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


soil_parameters = np.array([10.0, 0.1, 1.8]).reshape(-1, 3)
alpha_a, beta_a = 1.0, 5.0
learning_rate = 1e-3
fm = 47.6
bw = 20

bg_max, bg_min, s_max, s_min, lv_max, lv_min, ff_max, ff_min = data_max_min()
c = tf.placeholder(tf.float32, [None, 3])
c_n = (c-s_min)/(s_max-s_min)
lv_n, a = inverse_network(alpha_a, beta_a)
lv = lv_n*(lv_max-lv_min)+lv_min

p = tf.concat([lv_n, c_n], axis=1)
pre_bg_n = BPM(p)
pre_bg = (pre_bg_n*(bg_max-bg_min)+bg_min)/a

pre_ff_n = FFPM(lv_n)
pre_ff = pre_ff_n*(ff_max-ff_min)+ff_min

loss_s = pre_ff*a*a
loss_wb = (pre_bg[:, 1]+pre_bg[:, 0])/2/(pre_bg[:, 1]-pre_bg[:, 0])
loss_fm = tf.square((pre_bg[:, 1]+pre_bg[:, 0])/2-fm)
loss_wf = tf.maximum(bw-(pre_bg[:, 1]-pre_bg[:, 0]), 0)
loss = loss_s*loss_wb+(loss_fm+loss_wf)*1000
loss = tf.reshape(loss, [-1, 1])


#######  核心
t_vars = tf.trainable_variables()
updated_variables = [var for var in t_vars if var.name.startswith('inverse_design_network')]
optimizer = tf.train.AdamOptimizer(1e-1).minimize(loss, var_list=updated_variables)

restored_variables1 = [var for var in t_vars if var.name.startswith('pretrained_forward_modeling_network_for_out_of_plane_waves')]
restored_variables2 = [var for var in t_vars if var.name.startswith('MLP_for_filling_fraction')]
saver1 = tf.train.Saver(restored_variables1)
saver2 = tf.train.Saver(restored_variables2)
######  核心

loss_history = [100]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver1.restore(sess, 'pretrained_models/pretrained_forward_modeling_network_for_out_of_plane_waves.ckpt')
    saver2.restore(sess, 'pretrained_models/MLP_for_filling_fraction.ckpt')
    epoch = 0
    while True:
        _, loss_v = sess.run([optimizer, loss], feed_dict={c: soil_parameters})
        if loss_v <= np.mean(loss_history):
            d_lv, d_a = sess.run([lv, a])
            loss_history.append(loss_v)
            print(epoch, loss_v)
        if epoch > 100000:
            break
        epoch += 1


designed_topology = VAE_decoder(d_lv).reshape(-1, 50*50)
save_designed_results(designed_topology, d_a, 'results/optimized_results.xlsx')

















