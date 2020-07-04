# CollisionRiskprediction
Use LSTM to predict the satellites collision risk in the “future”.

欧洲航天局[卫星碰撞风险预测](https://kelvins.esa.int/collision-avoidance-challenge/home/)竞赛的代码。

### Environment

```python
Windows10
Python 3.7
Tensorflow 2.0.0
Pandas 1.0.0
Numpy 1.18.1
Sklearn 0.22.1

```

将train_data.csv放在当前文件夹中，然后运行train.py训练模型，运行predict.py预测模型。在logs文件夹中已有训练好的网络权重，可以直接运行predict.py

训练数据可以在这里下载：https://kelvins.esa.int/collision-avoidance-challenge/data/



请新建一个logs文件夹用于存放训练过程中网络的权重

