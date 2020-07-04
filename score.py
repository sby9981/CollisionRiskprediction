import numpy as np

def get_score(risk_true, risk_pre):
    assert len(risk_true) == len(risk_pre)
    tp, fp, fn = 0, 0, 0
    sum_square = 0
    n = 0
    beta = 2
    for i in range(len(risk_true)):   # 遍历数据得到tp、fn、fp的数量，并计算MSE的和
        if risk_true[i] > -6:
            if risk_pre[i] > -6:
                tp = tp+1
            else:
                fn = fn+1
            sum_square = sum_square + (risk_true[i] - risk_pre[i])**2    # 对MSE求和
            n = n+1
        else:
            if risk_pre[i] > -6:
                fp = fp+1

    # 计算各项指标
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    F = (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall)
    MSE = sum_square / n
    L = MSE / F
    print(MSE, precision, recall, F)    # 打印结果
    return L
