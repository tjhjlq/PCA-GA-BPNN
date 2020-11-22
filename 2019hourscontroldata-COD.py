# !/usr/bin/python3
# @File: 2019hourscontroldata-COD.py
# --coding:utf-8--
# @Author:Blazer
# @Time: 2020年08月03日18时52分03秒
# 说明:使用预处理后数据回归COD，全数据


from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model


# 开始计时
import time
start = time.process_time()

# 导入excel数据并通过dataframe生成dataframe
dataset_path = "E:/pycharm project/小论文/2019preprocesshourdata.xlsx"
column_names = ['index', 'date', 'time', 'volume1', 'volume2', 'volume',
                'CODi', 'pHi', 'NH3Ni', 'TNi', 'TPi',
                'CODet_1', 'NH3Net_1', 'TNet_1', 'TPet_1',
                'DOS1', 'DOS2', 'DOS3', 'ORPS1', 'ORPS2', 'MLSSS1', 'MLSSS2', 'interS1', 'interS2', 'interS3', 'interS4',
                'DON1', 'DON2', 'DON3', 'ORPN1', 'ORPN2', 'MLSSN', 'interN1', 'interN2', 'interN3', 'interN4', 'Totalwind',
                'DO11', 'DO12', 'DO13', 'ORP11', 'ORP12', 'MLSS1',
                'DO21', 'DO22', 'DO23', 'ORP21', 'ORP22', 'MLSS2',
                'DO31', 'DO32', 'DO33', 'ORP31', 'ORP32', 'MLSS3',
                'DO41', 'DO42', 'DO43', 'ORP41', 'ORP42', 'MLSS4',
                'wind1', 'wind2', 'wind3', 'wind4', 'wind5', 'mix1', 'mix2', 'mix3', 'mix4', 'mix5', 'mix6',
                'flocc1', 'flocc2', 'flocc3', 'flocc4', 'flocc5', 'flocc6', 'flocc7', 'flocc8',
                'filt1', 'filt2', 'filt3', 'filt4', 'filt5', 'filt6', 'filt7',
                'filt8', 'filt9', 'filt10', 'filt11', 'filt12', 'filt13', 'filt14',
                'CODe', 'NH3Ne', 'TNe', 'TPe'
                ]  # 其中MLSSN只有一个，wind4为四期反应池总风量，
data = pd.read_excel(dataset_path, names=column_names)

# 删除日期和时间和流量######################################
del data['index']
del data['date']
del data['time']
# del data['volume1']
# del data['volume2']  # 这里出于考虑到两个工艺对出水的影响比例，所以没有将两股分流量删去

# 删除整列为0的数据####################################
del data['interS1']
del data['interS4']
del data['interN1']
del data['interN3']
del data['interN4']

# 批量处理单独为0的数据
data = data.replace(0, 1e-8)  # 这一步太太关键了，不然就出不来

# 删除呈现差的数据行，看着图像找的
data.drop([29, 1978, 2008], axis=0, inplace=True)

# 出水指标需要取消语句//////////////////////////////////////////////////////////////////////
# CODe = data.pop('CODe')
NH3Ne = data.pop('NH3Ne')
TNe = data.pop('TNe')
TPe = data.pop('TPe')
# ///////////////////////////////////////////////////////////////////////////////////////

# 定义输出变量//////////////////////////////////////////////////////////////////////////////
output = 'CODe'
# data = pd.DataFrame(data, dtype=np.float)  # 这一步非常关键，因为在前面的操作中，各项相加将数据的类型改变了，要通过这个方式变回float
# //////////////////////////////////////////////////////////////////////////////////////////

# 取一部分进行训练
# data = data[1000:6000]

# 分割训练数据和测试数据
train_data = data.sample(frac=0.8, random_state=0)  # frac=0.8 means train data possess 80% of all
test_data = data.drop(train_data.index)

# 得到数据的描述性统计特征
train_stats = train_data.describe(include='all')
train_stats.pop(output)  # delete data what we need to predict and contract with the real one
train_stats = train_stats.transpose()

# 分离预测量
train_labels = train_data.pop(output)
test_labels = test_data.pop(output)
print(data)

# 导出标准化前数据csv/////////////////////////////////////////////////////////////////////////////////////////////////////
# test_data.to_csv("dataframe/raw_2019hoursdatawithcontrol-normedtestdata.csv", index=True)  # index 项决定是否包含head行
# train_data.to_csv("dataframe/raw_2019hoursdatawithcontrol-normedtraindata.csv", index=True)
# test_labels.to_csv("dataframe/raw_2019hoursdatawithcontrol-testlabels.csv", index=True)
# train_labels.to_csv("dataframe/raw_2019hoursdatawithcontrol-trainlabels.csv", index=True)
# ///////////////////////////////////////////////////////////////////////////////////////////////这个导出数据需要调整/////


# 归一化数据
def norm(x):
    return (x - train_stats['mean'])/train_stats['std']


normed_train_data = norm(train_data)
normed_test_data = norm(test_data)
print(normed_test_data)
testindex = normed_test_data.index
print(testindex)

# 导出标准化数据csv///////////////////////////////////////////////////////////////////////////////////////////////////////
# normed_test_data.to_csv("dataframe/2019hoursdatawithcontrol-normedtestdata.csv", index=True)  # index 项决定是否包含head行
# normed_train_data.to_csv("dataframe/2019hoursdatawithcontrol-normedtraindata.csv", index=True)
# test_labels.to_csv("dataframe/2019hoursdatawithcontrol-testlabels.csv", index=True)
# train_labels.to_csv("dataframe/2019hoursdatawithcontrol-trainlabels.csv", index=True)
# ///////////////////////////////////////////////////////////////////////////////////////////////这个导出数据需要调整/////


# 建立模型
def build_model():
    model = keras.Sequential([
        layers.Dense(20, activation=tf.nn.relu, input_shape=[len(train_data.keys())]),
        layers.Dense(40, activation=tf.nn.relu),
        layers.Dense(1)
    ])  # Watch out keys

    optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.5, epsilon=None, decay=0.0)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])  # MSE and MAE used in predict model
    return model


model = build_model()

# Inspect the model
# print(model.summary())

# Try out the model
# example_batch = normed_train_data[:5]
# example_result = model.predict(example_batch)
# print(example_result)


# 训练模型
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


EPOCHS = 100

history = model.fit(
    normed_train_data,
    train_labels,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[PrintDot()]  #
)

# Visualize the model's training progress
# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# print(hist.tail())


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(10, 6))
    ax1 = plt.subplot(1, 2, 1)  # 构造两个图
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error ' + output)
    plt.plot(hist['epoch'],
             hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'],
             hist['val_mean_absolute_error'],
             label='Val Error')
    # plt.ylim([0, 5])
    plt.legend()

    ax2 = plt.subplot(1, 2, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$' + output + '^2]')
    plt.plot(hist['epoch'],
             hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'],
             hist['val_mean_squared_error'],
             label='Val Error')
    # plt.ylim([0, 20])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.5, hspace=2)
    plt.legend()
    plt.show()


plot_history(history)

# EarlyStopping;The patience parameter is the amount of epochs to check for improvement
# model = build_model()
# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
#
# history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
#                     validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])
#
# print(plot_history(history))
# loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
# print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

# 做出预测
test_predictions = model.predict(normed_test_data).flatten()
print('\n', len(test_labels))
print(test_predictions)
error = test_predictions - test_labels
print(error)
print("最大误差为：", np.max(error))
error_p = sum(abs(error)/test_labels)/len(test_labels)
print('平均相对误差为：', error_p)


plt.figure(figsize=(10, 6))
ax1 = plt.subplot(1, 2, 1)
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [' + output + ']')
plt.ylabel('Predictions [' + output + ']')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.text(6, 4, error_p, fontdict={'size': '12', 'color': 'Crimson'})
plt.legend()


ax2 = plt.subplot(1, 2, 2)
plt.hist(error, bins=15)
plt.xlabel("Prediction Error [" + output + "]")
_ = plt.ylabel("Count")
plt.legend()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0.5, hspace=2)
plt.show()


# 画时序曲线图，预测数据和历史数据对比
all_labels = data.pop(output)
normed_all_data = norm(data)
pre_all_data = model.predict(normed_all_data).flatten()
n = len(model.predict(normed_all_data).flatten())

plt.figure(figsize=(10, 6))
x = np.arange(0, n, 1)

plt.plot(x,
         pre_all_data,
         linestyle='-',
         linewidth=2,
         color='#ff9999',
         marker=None,
         markersize=6,
         markeredgecolor='black',
         markerfacecolor='#ff9999',
         label='predict')

plt.plot(x,
         all_labels,
         linestyle='-',
         linewidth=2,
         color='steelblue',
         marker=None,
         markersize=6,
         markeredgecolor='black',
         markerfacecolor='steelblue',
         label='actual')

# 添加标题和坐标轴标签
plt.title('ANN model predict COD')
plt.xlabel('Time(hours)')
plt.ylabel('Effluent COD(mg/L)')

# 显示图例
plt.legend()

# 剔除图框上边界和右边界的刻度
plt.tick_params(top='off', right='off')
import datetime
plt.savefig('E:/pycharm project/小论文/结果图表/' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + 'COD-scheme1.jpg')
plt.show()

# 保存模型及权重
# model.save('history/2019hoursdatawithcontrol.h5')
# model.save_weights('history/2019hoursdatawithcontrol_weights.h5')

# 评估参数输出
from sklearn.metrics import r2_score
pd.set_option('display.max_columns', None)
hist = pd.DataFrame(history.history)
print(hist)
print("神经网络MAE=", np.array(hist.loc[99, ['mean_absolute_error']])[0])
print("神经网络RMSE=", np.sqrt(np.array(hist.loc[99, ['mean_squared_error']])[0]))
print("神经网络R2=", r2_score(test_predictions, test_labels))

# 记录时间
end = time.process_time()
print('Running time: %s Seconds' % (end-start))

