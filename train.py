import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
from dataread import DataReader
from net import MyRNN
import matplotlib.pyplot as plt

# 读取打乱后的数据
dataread = DataReader()
data_array, fin_risk = dataread.get_shuffle_data()

batchsz = 1024  # 批处理
val_size = 1800  # 验证集大小
train_db = tf.data.Dataset.from_tensor_slices((data_array[val_size:, :, :], fin_risk[val_size:]))   # 划分出训练集
val_db = tf.data.Dataset.from_tensor_slices((data_array[:val_size, :, :], fin_risk[:val_size]))     # 划分出验证集
train_db = train_db.batch(batchsz)
val_db = val_db.batch(batchsz)
print(train_db)  # ((None, 23, 102), (None,)), types: (tf.float64, tf.float64)>
print(val_db)


def main():
    units = 128    # LSTM网络参数量
    epochs = 150   # 训练轮数
    model = MyRNN(units)

    log_dir = "logs/"
    # 学习率下降，训练到一定轮数有助于更好的拟合数据
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',  # 参考值为测试集的损失值
        factor=0.8,          # 符合条件学习率降为原来的0.8倍
        min_delta=0.1,
        patience=10,         # 10轮测试集的损失值没有优化则下调学习率
        verbose=1
    )
    # 每30轮自动保存数据
    checkpoint_period = callbacks.ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss',
        save_weights_only=True,
        save_best_only=True,
        period=30
    )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.05,
        patience=20,
        verbose=1
    )

    # 设置训练参数，初始学习率为0.01，损失函数为MSE
    model.compile(optimizer=keras.optimizers.Adam(0.01),
                  loss='mse',
                  metrics=['mse'])

    # model.build(input_shape=(None, 25, 102))
    # model.load_weights(log_dir + 'last1.h5')     # 读取之前的权重继续训练

    # 训练模型
    history = model.fit(train_db, epochs=epochs, validation_data=val_db,
              callbacks=[reduce_lr, checkpoint_period])
    model.save_weights(log_dir + 'last1.h5')    # 保存最终权重
    model.summary()

    # 画出训练过程中训练集和验证集MSE的变化趋势
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()
