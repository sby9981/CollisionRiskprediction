
from net import MyRNN
from dataread import DataReader
from score import get_score


units = 128
model = MyRNN(units)             # 读取网络结构
model.build(input_shape=(None, 25, 102))
log_dir = "logs/"
model.load_weights(log_dir + 'last1.h5')      # 读取训练得到的网络权重

dataread = DataReader()
data_array, fin_risk, test_array, test_risk = dataread.get_data_array()
risk_pre = model.predict(test_array)    # 预测风险

L = get_score(test_risk, risk_pre)      # 通过get_score函数计算指标F
print(L)
