# -*- coding: utf-8 -*-

# 必要ライブラリをインポート
import tensorflow as tf
import numpy as np
import pickle

# 初期値を設定する。
length_of_side = 0.1 # 立方体の一片の長さ(m)
i_max = 10 # 電荷の最大値

# pickleファイルを読み込む。
dataset_elecs, dataset_fields = pickle.load(open('dataset20181219.pickle', 'rb'))
dataset_size = len(dataset_elecs)

"""
課題6:dataset_elecs, dataset_fieldsの規格化(0~1の値にする)を行う。
dataset_elecs[:, 0, :]をi_maxで割る。
dataset_elecs[:, 1:, :]をlength_of_sideで割る。
dataset_fieldsの要素全てを最小値で引き、その後最大値で割る。
"""
############## 課題6 ################
dataset_elecs[:, 0, :] /= i_max
dataset_elecs[:, 1:, :] /= length_of_side
dataset_fields -= np.min(dataset_fields)
dataset_fields /= np.max(dataset_fields)
#####################################

"""
課題7:dataset_elecs, dataset_fieldsをそれぞれfloat型に変換する。
その後(10000, 2 * 3), (10000, 11 * 11 * 3 * 6)にreshapeする。
"""
############## 課題7 ################
dataset_elecs = np.array(dataset_elecs, dtype=np.float32)
dataset_fields = np.array(dataset_fields, dtype=np.float32)
dataset_elecs = np.reshape(dataset_elecs, (dataset_size, 2 * 3))
dataset_fields = np.reshape(dataset_fields, (dataset_size, 11 * 11 * 3 * 6))
#####################################

"""
課題8:データセットのうち1000をテストデータ、残りを学習用にする。
学習用データをelecs_train, fields_train
テストデータをelecs_test, fields_test
という変数名とする。
"""
############## 課題8 ################
elecs_train = dataset_elecs[:dataset_size-1000, :]
elecs_test = dataset_elecs[dataset_size-1000:, :]
fields_train = dataset_fields[:dataset_size-1000, :]
fields_test = dataset_fields[dataset_size-1000:, :]
#####################################

tf.reset_default_graph()

# graphの構築
# placeholderの定義
x = tf.placeholder("float", shape=[None, 11 * 11 * 3 * 6])
y_ = tf.placeholder("float", shape=[None, 6])

# 荷重作成
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# バイアス作成
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 畳み込み処理を定義
def conv2d_pad(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# プーリング処理を定義
def max_pool_2_2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

"""
課題9:tensorflowグラフを構築する。
以下の実装例でも動作しますが、
時間があれば自身でフィルタの大きさ、特徴マップの大きさ、隠れ層の大きさ、畳み込み層を増やすなど工夫してください。
"""
############## 課題9 ################
# 畳み込み層1
W_conv1 = weight_variable([3, 3, 18, 256])
b_conv1 = bias_variable([256])
x_image = tf.reshape(x, [-1, 11, 11, 18])
h_conv1 = tf.nn.relu(conv2d_pad(x_image, W_conv1) + b_conv1)
# プーリング層1
h_pool1 = max_pool_2_2(h_conv1)

# 畳み込み層2
W_conv2 = weight_variable([3, 3, 256, 256])
b_conv2 = bias_variable([256])
h_conv2 = tf.nn.relu(conv2d_pad(h_pool1, W_conv2) + b_conv2)
# プーリング層2
h_pool2 = max_pool_2_2(h_conv2)

# 全結合層1
W_fc1 = weight_variable([256*3*3, 1024])
b_fc1 = bias_variable([1024])
h_flat = tf.reshape(h_pool2, [-1, 256*3*3])
h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

# ドロップアウト
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 全結合層2
W_fc2 = weight_variable([1024, 6])
b_fc2 = bias_variable([6])
y_out = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# 学習誤差を求める
loss = tf.reduce_mean(tf.square(y_ - y_out))

# 最適化処理
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
#####################################

# セッション開始
with tf.Session() as sess:
    # グラフの保存クラス
    saver = tf.train.Saver()
    # 学習済みグラフの読み込み
    saver.restore(sess, "./20181219model")
    
    """
    課題11:fields_test, elecs_testからbatch, outputを作成、y_outを呼び出し、outputと共に出力する。
    """
    ############## 課題11 ################
    batch = fields_test
    output = elecs_test
    prediction = y_out.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
    print("output" + str(output[1, :]))
    print("prediction" + str(prediction[1, :]))
    #####################################