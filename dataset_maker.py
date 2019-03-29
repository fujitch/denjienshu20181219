# -*- coding: utf-8 -*-
# 必要なライブラリのインポート
import math
import numpy as np
import random
import pickle
"""
課題1:初期値を設定する。
"""
############## 課題1 ################
length_of_side = 0.1 # 立方体の一片の長さ(m)
i_max = 10 # 電流の最大値
pi = math.pi # 円周率
num_of_dataset = 10000 # データセットの数
####################################

# x, y, z座標のリストを作成
x_coordinates = []
for x in range(11):
    x_coordinates.append(x * length_of_side / 10)
y_coordinates = []
for y in range(11):
    y_coordinates.append(y * length_of_side / 10)
z_coordinates = []
for z in range(11):
    z_coordinates.append(z * length_of_side / 10)
        
# 無限電流を配置し、磁場を作成する関数を定義
def make_elec_field():
    """
    課題2:無限電流となる変数elecsを定義し、(3, num_of_elecs)のゼロ行列とする。
    0行目0列目:0~i_maxの範囲でランダムに電流値を挿入
    1行目0列目:y_coordinatesの中で両端を除いた座標を挿入
    2行目0列目:z_coordinatesの中で両端を除いた座標を挿入
    0行目1列目:0~i_maxの範囲でランダムに電流値を挿入
    1行目1列目:z_coordinatesの中で両端を除いた座標を挿入
    2行目1列目:x_coordinatesの中で両端を除いた座標を挿入
    """
    ############## 課題2 ################
    elecs = np.zeros((3, 2))
    elecs[0, 0] = random.random() * i_max
    elecs[1, 0] = y_coordinates[random.randint(1, len(y_coordinates)-2)]
    elecs[2, 0] = z_coordinates[random.randint(1, len(z_coordinates)-2)]
    elecs[0, 1] = random.random() * i_max
    elecs[1, 1] = z_coordinates[random.randint(1, len(z_coordinates)-2)]
    elecs[2, 1] = x_coordinates[random.randint(1, len(x_coordinates)-2)]
    ####################################
    
    """
    課題3:磁場となる変数fieldsを定義し、(11, 11, 3, 6)のゼロ行列とする。
    11, 11:立方体の１面の座標行列
    3:磁場ベクトルxyz成分
    6:立方体6面
    をそれぞれ表している。
    """
    ############## 課題3 ################
    fields = np.zeros((11, 11, 3, 6))
    ####################################
    
    # x軸平行電流の作る磁場を挿入
    elec = elecs[:, 0]
    # xy平面(z=0)
    z = z_coordinates[0]
    for y_index in range(len(y_coordinates)):
        y = y_coordinates[y_index]
        r = np.array([y - elec[1], z - elec[2]])
        r_rotate = np.array([-r[1], r[0]])
        fields[:, y_index, 1, 0] += (elec[0] * r_rotate[0]) / (2 * pi * pow(np.linalg.norm(r_rotate), 2))
        fields[:, y_index, 2, 0] += (elec[0] * r_rotate[1]) / (2 * pi * pow(np.linalg.norm(r_rotate), 2))
    # xy平面(z=0.1)
    z = z_coordinates[10]
    for y_index in range(len(y_coordinates)):
        y = y_coordinates[y_index]
        r = np.array([y - elec[1], z - elec[2]])
        r_rotate = np.array([-r[1], r[0]])
        fields[:, y_index, 1, 1] += (elec[0] * r_rotate[0]) / (2 * pi * pow(np.linalg.norm(r_rotate), 2))
        fields[:, y_index, 2, 1] += (elec[0] * r_rotate[1]) / (2 * pi * pow(np.linalg.norm(r_rotate), 2))
    # zx平面(y=0)
    y = y_coordinates[0]
    for z_index in range(len(z_coordinates)):
        z = z_coordinates[z_index]
        r = np.array([y - elec[1], z - elec[2]])
        r_rotate = np.array([-r[1], r[0]])
        fields[z_index, :, 1, 4] += (elec[0] * r_rotate[0]) / (2 * pi * pow(np.linalg.norm(r_rotate), 2))
        fields[z_index, :, 2, 4] += (elec[0] * r_rotate[1]) / (2 * pi * pow(np.linalg.norm(r_rotate), 2))
    # zx平面(y=0.1)
    y = y_coordinates[10]
    for z_index in range(len(z_coordinates)):
        z = z_coordinates[z_index]
        r = np.array([y - elec[1], z - elec[2]])
        r_rotate = np.array([-r[1], r[0]])
        fields[z_index, :, 1, 5] += (elec[0] * r_rotate[0]) / (2 * pi * pow(np.linalg.norm(r_rotate), 2))
        fields[z_index, :, 2, 5] += (elec[0] * r_rotate[1]) / (2 * pi * pow(np.linalg.norm(r_rotate), 2))
    """
    課題4:上記スクリプトを参考にy軸平行電流の作る磁場を挿入してください。
    自分で書きたい場合は上記を消して実装してください。
    """
    ############## 課題4 ################
    # y軸平行電流の作る磁場を挿入
    elec = elecs[:, 1]
    # xy平面(z=0)
    z = z_coordinates[0]
    for x_index in range(len(x_coordinates)):
        x = x_coordinates[x_index]
        r = np.array([z - elec[1], x - elec[2]])
        r_rotate = np.array([-r[1], r[0]])
        fields[x_index, :, 2, 0] += (elec[0] * r_rotate[0]) / (2 * pi * pow(np.linalg.norm(r_rotate), 2))
        fields[x_index, :, 0, 0] += (elec[0] * r_rotate[1]) / (2 * pi * pow(np.linalg.norm(r_rotate), 2))
    # xy平面(z=0.1)
    z = z_coordinates[10]
    for x_index in range(len(x_coordinates)):
        x = x_coordinates[x_index]
        r = np.array([z - elec[1], x - elec[2]])
        r_rotate = np.array([-r[1], r[0]])
        fields[x_index, :, 2, 1] += (elec[0] * r_rotate[0]) / (2 * pi * pow(np.linalg.norm(r_rotate), 2))
        fields[x_index, :, 0, 1] += (elec[0] * r_rotate[1]) / (2 * pi * pow(np.linalg.norm(r_rotate), 2))
    # yz平面(x=0)
    x = x_coordinates[0]
    for z_index in range(len(z_coordinates)):
        z = z_coordinates[z_index]
        r = np.array([z - elec[1], x - elec[2]])
        r_rotate = np.array([-r[1], r[0]])
        fields[:, z_index, 2, 2] += (elec[0] * r_rotate[0]) / (2 * pi * pow(np.linalg.norm(r_rotate), 2))
        fields[:, z_index, 0, 2] += (elec[0] * r_rotate[1]) / (2 * pi * pow(np.linalg.norm(r_rotate), 2))
    # yz平面(x=0.1)
    x = x_coordinates[10]
    for z_index in range(len(z_coordinates)):
        z = z_coordinates[z_index]
        r = np.array([z - elec[1], x - elec[2]])
        r_rotate = np.array([-r[1], r[0]])
        fields[:, z_index, 2, 3] += (elec[0] * r_rotate[0]) / (2 * pi * pow(np.linalg.norm(r_rotate), 2))
        fields[:, z_index, 0, 3] += (elec[0] * r_rotate[1]) / (2 * pi * pow(np.linalg.norm(r_rotate), 2))
    ####################################
    return elecs, fields

"""
課題5:ゼロ行列dataset_elecs, dataset_fieldsを定義する。
それぞれ、(num_of_dataset, 3, 2), (num_of_dataset, 11, 11, 3, 6)のゼロ行列とする。
make_elec_field関数をnum_of_dataset回ループし、elecsとfieldsをそれぞれdataset_elecs, dataset_fieldsに挿入していく。
dataset_elecs, dataset_fieldsをtuple型としてdatasetという変数に挿入する。
dataset変数をpickleファイルとしてローカルストレージに保存する。ファイル名は任意。
"""
############## 課題5 ################
dataset_elecs = np.zeros((num_of_dataset, 3, 2))
dataset_fields = np.zeros((num_of_dataset, 11, 11, 3, 6))
for i in range(num_of_dataset):
    elecs, fields = make_elec_field()
    dataset_elecs[i, :, :] = elecs
    dataset_fields[i, :, :, :, :] = fields
dataset = (dataset_elecs, dataset_fields)
pickle.dump(dataset, open("dataset20181219.pickle", "wb"))
####################################