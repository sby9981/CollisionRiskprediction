import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing


class DataReader():
    df = None
    train_stats = None
    data_array = None
    fin_risk = None

    # 数据读取与初步处理
    def pre_data(self):
        df = pd.read_csv("train_data.csv")  # [162634, 103]
        encoder = preprocessing.LabelEncoder()
        df['c_object_type'] = encoder.fit_transform(df['c_object_type'])      # 对非数字参量进行编码
        df.fillna(value=0, inplace=True)
        self.df = df
        return df

    # 统计每个event的final risk和最小time to tca
    def get_risk(self):
        df = self.pre_data()
        fin_risk = np.zeros(13154)        # 统计每个event的final risk用于训练和测试
        fin_timetotca = np.zeros(13154)       # 统计每个event的最小time to tca，为后续筛选数据做准备
        event_id = 0
        for index, row in df.iterrows():        # for循环逐行读取数据
            if row['event_id'] == event_id:     # 每个event独自保存
                fin_risk[event_id] = row['risk']
                fin_timetotca[event_id] = row['time_to_tca']
            else:
                event_id = event_id + 1
                fin_risk[event_id] = row['risk']
                fin_timetotca[event_id] = row['time_to_tca']
        self.fin_risk = fin_risk
        self.fin_timetotca = fin_timetotca
        return fin_risk, fin_timetotca

    # 得到每个参量的平均值和方差，为归一化做准备
    def get_stats(self):
        df = self.pre_data()
        event = df.pop('event_id')
        risk = df.pop('risk')
        train_stats = df.describe()
        self.train_stats = train_stats.transpose()
        df['event_id'] = event
        df['risk'] = risk
        return self.train_stats

    # 归一化
    def norm(self, x):
        train_stats = self.get_stats()
        return (x - train_stats['mean']) / train_stats['std']

    def get_norm_data(self):
        self.get_stats()
        df = self.df
        event = df.pop('event_id')
        risk = df.pop('risk')
        df = self.norm(df)
        df['risk'] = risk
        df['event_id'] = event
        return df

    # 筛选数据，并划分为训练集+验证集和测试集
    def get_data_array(self):
        df = self.get_norm_data()
        fin_risk, final_timetotca = self.get_risk()
        final_risk = np.zeros(7293)     # 经过运行下面代码，打印data_array.shape得到的训练集+验证集包含event数量为7293
        test_risk = np.zeros(1902)      # 测试集取time to tca>2时，测试集包含event数量为1902（取time to tca>1时数量为2047）
        data_array = []
        test_array = []
        temp_vector = []
        event_id = 0
        n = 0
        k = 0
        k1 = 0
        for index, row in df.iterrows():
            if row['event_id'] < 3000:       # 取测试集，通过改变该条件可以得到不同的测试集
                if row['event_id'] == event_id:
                    if final_timetotca[event_id] < 1:      # 筛选最小time to tca<1的数据，这样对final risk的估计才更准确
                        if row['time_to_tca'] > -0.671:        # 2经过归一化后对应-0.671,1对应-1.169
                            temp_vector.append(row.values[:-1])
                            n = n + 1
                else:
                    if n > 0:   # 若该event有数据被保存
                        if n < 25:
                            for i in range(25 - n):      # 数据维度统一化，填充全0行到25行
                                temp_vector.append(np.zeros(102, ))
                        test_array.append(np.array(temp_vector))
                        test_risk[k1] = fin_risk[event_id]    # 保存final risk
                        k1 = k1+1
                    temp_vector = []
                    event_id = event_id + 1
                    n = 0
                    if final_timetotca[event_id] < 1:   # 下一个event的第一行
                        if row['time_to_tca'] > -0.671:
                            temp_vector.append(row.values[:-1])
                            n = n + 1
            else:
                if row['event_id'] == event_id:     # 训练集+验证集，数据筛选方式与测试集同理
                    if final_timetotca[event_id] < 1:
                        temp_vector.append(row.values[:-1])
                        n = n + 1
                else:
                    if n > 0:
                        if n < 25:
                            for i in range(25 - n):
                                temp_vector.append(np.zeros(102, ))
                        data_array.append(np.array(temp_vector))
                        final_risk[k] = fin_risk[event_id]
                        k = k+1
                    temp_vector = []
                    event_id = event_id + 1
                    n = 0
                    if final_timetotca[event_id] < 1:
                        temp_vector.append(row.values[:-1])
                        n = n + 1

        test_array = np.array(test_array)
        data_array = np.array(data_array)
        print(data_array.shape,  test_array.shape)
        return data_array, final_risk, test_array, test_risk

    # 打乱训练集和验证集的分布
    def get_shuffle_data(self):
        data_array, fin_risk, test_array, test_risk = self.get_data_array()
        index = np.array(range(7293))
        np.random.shuffle(index)
        data_array = data_array[index]
        fin_risk = fin_risk[index]
        return data_array, fin_risk

