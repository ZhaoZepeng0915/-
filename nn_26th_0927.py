import numpy as np
import pandas as pd

class NeuralNetwork():
    def __init__(self):
        self.df_prepare = pd.DataFrame()
        # 设置节点数和学习率
        self.inodes = 13
        self.hnodes = 13
        self.onodes = 2
        self.learningrate = 0.5
        # 定义sigmoid函数
        self.s_function = lambda x:  1/(1 + np.exp(-x))
        # 初始化w_ih矩阵和w_ho矩阵，随机正态分布，均值为0，标准差为1/(self.hnodes ** (-0.5))
        self.w_ih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.w_ho = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))


    # 数据处理
    def data_prepare(self, raw_data):
        df = raw_data
        '''
            对数据进行预处理，去掉有缺失值的行;创建新的索引
            创建新的两列，分别为日期和时间列
        '''
        df.dropna(axis=0, how='all', inplace=True)
        df.fillna(method="ffill", inplace=True)
        df = df.reset_index(drop=True)
        df['create_time'] = pd.to_datetime(df['create_time'], errors='coerce')
        # 生成日期列、时间列
        df['date'] = df['create_time'].dt.date
        df['h-m-s'] = df['create_time'].dt.time

        # 删除8:30之前和18:00之后的数据
        indexNames = df[(df['h-m-s'] < datetime.time(8, 30, 0, 0)) | (df['h-m-s'] > datetime.time(18, 0, 0, 0))].index
        df.drop(indexNames, inplace=True)

        # 计算总送风量和冷量，并将所有参数处理到（0,1）的范围内
        def sum_columns(columns):
            label_1 = columns
            label_2 = list(df[label_1].sum(axis=1))  # 按照列相加
            return label_2

        df['总送风量'] = sum_columns(['v5586_VAV72602送风量', 'v5587_VAV72603送风量', 'v5588_VAV72604送风量',
                                  'v5589_VAV72605送风量', 'v5590_VAV72606送风量', 'v5591_VAV72607送风量',
                                  'v5592_VAV72608送风量', 'v5593_VAV72609送风量', 'v5594_VAV72610送风量',
                                  'v5595_VAV72611送风量', 'v5596_VAV72612送风量', 'v5597_VAV72613送风量'])
        df['总冷量'] = list(1.08 * df['总送风量'] * (0.3 * df['v0001_室外温度'] + 0.7 * df['v0036_回风温度'] - df['v0038_送风温度']))
        df['总送风量_to1'] = list(df['总送风量']/2500)
        df['总冷量_to1'] = list(df['总冷量'] / 30000)

        # 归一化需要作为输入的温度
        df['']
        self.df_prepare = df
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
        self.w_ho += self.learningrate * np.dot((errors_output * final_outputs * (1-final_outputs)),
                                                np.transpose(hidden_outputs))
        # 更新权重w_ih+= △w_ih
        self.w_ih += self.learningrate * np.dot((errors_hidden * hidden_outputs * (1 - hidden_outputs)),
                                                np.transpose(inputs))


    # 查询
    def query(self, inputs_list):
        # 将输入的列表转换为2维数组
        inputs = np.array(inputs_list, ndmin=2).T
        # 计算隐藏层的信号
        hidden_inputs = np.dot(self.w_ih,inputs)
        hidden_outputs = self.s_function(hidden_inputs)
        # 计算输出层的信号
        final_inputs = np.dot(self.w_ho, hidden_outputs)
        final_outputs = self.s_function(final_inputs)

        return final_outputs

    # 损失函数
    def mse_loss(y_tr, y_pre):
        return ((y_tr - y_pre) ** 2).mean()


if __name__ == "__main__":
    # 读取数据
    data = pd.read_csv(r'原始数据\26-27的数据.csv', low_memory=False)
    aaa = NeuralNetwork()
    # 数据处理
    aaa.df_prepare(data)
    # 模型训练

    data_train = []
    label_train = []



    aaa.train(data_train, label_train)

    # 预测送风量和冷量
    inputs = []
    result = aaa.query(inputs)
    print(result)
