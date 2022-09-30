import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
class NeuralNetwork():
    def __init__(self):
        self.df_prepare = pd.DataFrame()
        # 设置节点数和学习率
        self.inodes = 22
        self.hnodes = 22
        self.onodes = 2
        self.learningrate = 0.5
        # 定义sigmoid函数
        self.s_function = lambda x:  1/(1 + 2.718**(-x))
        # 初始化w_ih矩阵和w_ho矩阵，随机正态分布，均值为0，标准差为1/(self.hnodes ** (-0.5))
        self.w_ih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.w_ho = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))


    # 数据处理
    def data_prepare(self, raw_data1,raw_data2):
        df1 = raw_data1
        df2 = raw_data2
        '''
            对数据进行预处理，去掉有缺失值的行;创建新的索引
            创建新的两列，分别为日期和时间列
        '''
        df1.dropna(axis=0, how='all', inplace=True)
        df1.fillna(method="ffill", inplace=True)
        df1 = df1.reset_index(drop=True)
        df1['create_time'] = pd.to_datetime(df1['create_time'], errors='coerce')
        # 生成日期列、时间列
        df1['date'] = df1['create_time'].dt.date
        df1['h-m-s'] = df1['create_time'].dt.time

        df2.dropna(axis=0, how='all', inplace=True)
        df2.fillna(method="ffill", inplace=True)
        df2 = df2.reset_index(drop=True)
        df2['create_time'] = pd.to_datetime(df2['create_time'], errors='coerce')
        # 生成日期列、时间列
        df2['date'] = df2['create_time'].dt.date
        df2['h-m-s'] = df2['create_time'].dt.time
        # 删除8:00之前和18:00之后的数据
        indexNames = df1[(df1['h-m-s'] < datetime.time(8, 0, 0, 0)) | (df1['h-m-s'] > datetime.time(18, 0, 0, 0))].index
        df1.drop(indexNames, inplace=True)
        df1 = df1.reset_index(drop=True)
        df2.drop(indexNames, inplace=True)
        df2 = df2.reset_index(drop=True)

        # 将时间序列化为整数后作为输入
        def time_to_int(t):
            t1 = str(t)
            h, m, s = t1.strip().split(":")
            return int(h) * 3600 + int(m) * 60 + int(s)

        df1['h-m-s-int'] = list(map(time_to_int, df1['h-m-s']))

        # 计算总送风量和冷量
        def sum_columns(columns):
            label_1 = columns
            label_2 = list(df1[label_1].sum(axis=1))  # 按照列相加
            return label_2
        df1['总送风量'] = sum_columns(['v5586_VAV72602送风量', 'v5587_VAV72603送风量', 'v5588_VAV72604送风量',
                                  'v5589_VAV72605送风量', 'v5590_VAV72606送风量', 'v5591_VAV72607送风量',
                                  'v5592_VAV72608送风量', 'v5593_VAV72609送风量', 'v5594_VAV72610送风量',
                                  'v5595_VAV72611送风量', 'v5596_VAV72612送风量', 'v5597_VAV72613送风量'])
        # 训练模型需要的输出列
        df1['风机功率'] = df2['tag_value'].copy(deep = True)
        df1['总冷量'] = list(1.08 * df1['总送风量'] * (0.3 * df1['v0001_室外温度'] + 0.7 * df1['v0036_回风温度'] - df1['v0038_送风温度']))

        # 归一化所有需要作为输入的参数,共22个参数
        df1['hms_to_1'] = list(df1['h-m-s-int'] / 80000)
        df1['风机功率_to1'] = list(df1['风机功率'] / 5)
        df1['总冷量_to1'] = list(df1['总冷量'] / 30000)
        df1['室外温度_to1'] = list(map(lambda x: (x - 20.0)/20, df1['v0001_室外温度']))
        df1['vav02_温度设定值_to1'] = list(map(lambda x: (x - 20.0) / 20, df1['v4844_VAV72602设定温度']))
        df1['vav04_温度设定值_to1'] = list(map(lambda x: (x - 20.0) / 20, df1['v4846_VAV72604设定温度']))
        df1['vav05_温度设定值_to1'] = list(map(lambda x: (x - 20.0) / 20, df1['v4847_VAV72605设定温度']))
        df1['vav06_温度设定值_to1'] = list(map(lambda x: (x - 20.0) / 20, df1['v4848_VAV72606设定温度']))
        df1['vav07_温度设定值_to1'] = list(map(lambda x: (x - 20.0) / 20, df1['v4849_VAV72607设定温度']))
        df1['vav08_温度设定值_to1'] = list(map(lambda x: (x - 20.0) / 20, df1['v4850_VAV72608设定温度']))
        df1['vav10_温度设定值_to1'] = list(map(lambda x: (x - 20.0) / 20, df1['v4852_VAV72610设定温度']))
        df1['vav12_温度设定值_to1'] = list(map(lambda x: (x - 20.0) / 20, df1['v4854_VAV72612设定温度']))
        df1['vav13_温度设定值_to1'] = list(map(lambda x: (x - 20.0) / 20, df1['v4855_VAV72613设定温度']))

        df1['vav02_房间温度_to1'] = list(map(lambda x: (x - 20.0) / 20, df1['v0068_VAV72602房间温度']))
        df1['vav04_房间温度_to1'] = list(map(lambda x: (x - 20.0) / 20, df1['v0070_VAV72604房间温度']))
        df1['vav05_房间温度_to1'] = list(map(lambda x: (x - 20.0) / 20, df1['v0071_VAV72605房间温度']))
        df1['vav06_房间温度_to1'] = list(map(lambda x: (x - 20.0) / 20, df1['v0072_VAV72606房间温度']))

        df1['vav07_房间温度_to1'] = list(map(lambda x: (x - 20.0) / 20, df1['v0073_VAV72607房间温度']))
        df1['vav08_房间温度_to1'] = list(map(lambda x: (x - 20.0) / 20, df1['v0074_VAV72608房间温度']))
        df1['vav10_房间温度_to1'] = list(map(lambda x: (x - 20.0) / 20, df1['v0076_VAV72610房间温度']))
        df1['vav12_房间温度_to1'] = list(map(lambda x: (x - 20.0) / 20, df1['v0078_VAV72612房间温度']))
        df1['vav13_房间温度_to1'] = list(map(lambda x: (x - 20.0) / 20, df1['v0079_VAV72613房间温度']))

        self.df_prepare = df1

        return self.df_prepare


    # 训练模型
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # 计算隐藏层的信号
        hidden_inputs = np.dot(self.w_ih, inputs)
        hidden_outputs = self.s_function(hidden_inputs)
        # 计算输出层的信号
        final_inputs = np.dot(self.w_ho, hidden_outputs)
        final_outputs = self.s_function(final_inputs)

        # 计算误差
        errors_output = targets - final_outputs
        errors_hidden = np.dot(self.w_ho.T, errors_output)
        # 更新权重w_ho+= △w_ho
        self.w_ho = self.w_ho + self.learningrate * np.dot((errors_output * final_outputs * (1-final_outputs)),
                                                np.transpose(hidden_outputs))
        # 更新权重w_ih+= △w_ih
        self.w_ih = self.w_ih + self.learningrate * np.dot((errors_hidden * hidden_outputs * (1 - hidden_outputs)),
                                                np.transpose(inputs))


    # 查询
    def query(self, inputs_list):
        # 将输入的列表转换为2维数组
        inputs = np.array(inputs_list, ndmin=2).T
        # 计算隐藏层的信号
        hidden_inputs = np.dot(self.w_ih, inputs)
        hidden_outputs = self.s_function(hidden_inputs)
        # 计算输出层的信号
        final_inputs = np.dot(self.w_ho, hidden_outputs)
        final_outputs = self.s_function(final_inputs)

        return final_outputs

    # 计算误差，输出计算结果
    def accuracy_test(self):
        df = aaa.data_prepare(data_raw1, data_raw2)
        err_ahu_power = []
        err_cool_cap = []
        err_rate_air = []
        err_rate_cool = []
        for i in range(1200, 1700, 1):
            input_test = list(df.iloc[i, -22:])
            label_test = list(df.iloc[i, -21:-19])
            y_test = aaa.query(input_test)
            y_true = np.array(label_test, ndmin=2).T
            error_t = y_test - y_true
            accuracy_test = np.squeeze(error_t, axis=None).tolist()
            # 将百分比换算回实际物理值
            err_ahu_power.append(accuracy_test[0] * 5)
            err_cool_cap.append(accuracy_test[1] * 30000)
            # 误差百分比
            error_rate = (y_test - y_true)/y_true
            accuracy_rate = np.squeeze(error_rate, axis=None).tolist()
            err_rate_air.append(accuracy_rate[0])
            err_rate_cool.append(accuracy_rate[1])

        # 可视化结果
        # 构造多个ax,共享x轴
        x = range(500)
        fig, ax1 = plt.subplots(figsize=(9, 6))
        ax3 = ax1.twinx()
        # 绘制
        img1, = ax1.plot(x, err_ahu_power, c='tab:blue')
        img3, = ax3.plot(x, err_rate_air, c='r', linestyle='--')
        # 设置其他细节
        ax1.set_xlabel('time', fontsize=14)
        ax1.set_ylim(-1.5, 1.5)
        ax3.set_ylim(-0.5, 0.5)

        fig, ax2 = plt.subplots(figsize=(9, 6))
        ax4 = ax2.twinx()
        # 绘制
        img2, = ax2.plot(x, err_cool_cap, c='tab:orange')
        img4, = ax4.plot(x, err_rate_cool, c='r', linestyle='--')

        # 设置其他细节
        ax2.set_xlabel('time', fontsize=14)
        ax2.set_ylim(-2000, 8000)
        ax4.set_ylim(-0.2, 0.5)

        plt.show()
        return None


if __name__ == "__main__":
    # 读取数据
    aaa = NeuralNetwork()
    data_raw1 = pd.read_csv(r'zongbu_2022-09-26_2022-09-28.csv', low_memory=False)
    data_raw2 = pd.read_csv(r'power_2022-09-26_2022-09-28.csv', low_memory=False)
    # 数据处理
    df1 = aaa.data_prepare(data_raw1, data_raw2)
    # 用前两天的数据进行模型训练
    for i in range(1, 1200, 1):
        data_train = list(df1.iloc[i, -22:])
        # 将一分钟后的数据作为预测的label
        label_train = df1.iloc[i+1, -21:-19]
        aaa.train(data_train, label_train)
    # 检测训练模型的准确性,输出计算结果
    aaa.accuracy_test()
    # 给定输入值，预测送风量和冷量
    # inputs = [26,22,22,26,20,22,20,20,20,20,20,20,20]
    # result = aaa.query(inputs)
    # print(result)
