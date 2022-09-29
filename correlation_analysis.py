import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import matplotlib as mpt
from matplotlib import pyplot as plt


from IPython.core.interactiveshell import InteractiveShell

plt.rcParams['font.family'] = 'SimHei'  # 防止画图时中文乱码
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
InteractiveShell.ast_node_interactivity = 'all'
'''
分析输入和输出之间的线性相关性

'''


class CorrAnaly():
    def __init__(self):
        self.df1_prepare = pd.DataFrame()
        self.df2_prepare = pd.DataFrame()

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
        df2.dropna(axis=0, how='all', inplace=True)
        df2.fillna(method="ffill", inplace=True)
        df2 = df2.reset_index(drop=True)

        # 生成日期列、时间列
        df1['create_time'] = pd.to_datetime(df1['create_time'], errors='coerce')
        df1['date'] = df1['create_time'].dt.date
        df1['h-m-s'] = df1['create_time'].dt.time
        df2['create_time'] = pd.to_datetime(df1['create_time'], errors='coerce')
        df2['date'] = df2['create_time'].dt.date
        df2['h-m-s'] = df2['create_time'].dt.time

        # 删除8:30之前和18:00之后的数据
        indexNames1 = df1[(df1['h-m-s'] < datetime.time(8, 30, 0, 0)) | (df1['h-m-s'] > datetime.time(18, 0, 0, 0))].index
        df1.drop(indexNames1, inplace=True)
        indexNames2 = df2[
            (df2['h-m-s'] < datetime.time(8, 30, 0, 0)) | (df2['h-m-s'] > datetime.time(18, 0, 0, 0))].index
        df2.drop(indexNames2, inplace=True)


        # 计算总送风量和冷量
        def sum_columns(columns):
            label_1 = columns
            label_2 = list(df1[label_1].sum(axis=1))  # 按照列相加
            return label_2

        df1['总送风量'] = sum_columns(['v5586_VAV72602送风量', 'v5587_VAV72603送风量', 'v5588_VAV72604送风量',
                                  'v5589_VAV72605送风量', 'v5590_VAV72606送风量', 'v5591_VAV72607送风量',
                                  'v5592_VAV72608送风量', 'v5593_VAV72609送风量', 'v5594_VAV72610送风量',
                                  'v5595_VAV72611送风量', 'v5596_VAV72612送风量', 'v5597_VAV72613送风量'])
        df1['总冷量'] = list(1.08 * df1['总送风量'] * (0.3 * df1['v0001_室外温度'] + 0.7 * df1['v0036_回风温度'] - df1['v0038_送风温度']))
        df2['送风机功率'] = df2['tag_value']
        self.df1_prepare = df1
        self.df2_prepare = df2

        return self.df1_prepare, self.df2_prepare

    def data_heatmap(self):
        df = pd.DataFrame()
        data1 = self.df1_prepare
        data2 = self.df2_prepare
        df['date'] = data1['create_time'].dt.date.copy(deep=True)
        df['h-m-s'] = data1['create_time'].dt.time.copy(deep=True)

        # 将时间序列化为整数后作为输入
        def time_to_int(t):
            t1 = str(t)
            h, m, s = t1.strip().split(":")
            return int(h) * 3600 + int(m) * 60 + int(s)
        df['h-m-s-int'] = list(map(time_to_int, df['h-m-s']))

        list1 = ['date', 'h-m-s', 'v4844_VAV72602设定温度', 'v4845_VAV72603设定温度', 'v4846_VAV72604设定温度', 'v4847_VAV72605设定温度',
                 'v4848_VAV72606设定温度', 'v4849_VAV72607设定温度', 'v4850_VAV72608设定温度', 'v4851_VAV72609设定温度',
                 'v4852_VAV72610设定温度', 'v4853_VAV72611设定温度', 'v4854_VAV72612设定温度', 'v4855_VAV72613设定温度',
                 'v0068_VAV72602房间温度', 'v0069_VAV72603房间温度', 'v0070_VAV72604房间温度', 'v0071_VAV72605房间温度',
                 'v0072_VAV72606房间温度', 'v0073_VAV72607房间温度', 'v0074_VAV72608房间温度', 'v0075_VAV72609房间温度',
                 'v0076_VAV72610房间温度', 'v0077_VAV72611房间温度', 'v0078_VAV72612房间温度', 'v0079_VAV72613房间温度',
                 'v0001_室外温度']
        for i in list1:
            df[i] = data1[i].copy(deep=True)
        df['送风机功率'] = data2['送风机功率'].copy(deep=True)
        df['总冷量'] = data1['总冷量']
        print(df.head(20))


        # 相关系数表
        corr_df = df.corr()

        # 温度、湿度相关系数图
        fig, ax = plt.subplots(figsize=(8, 15), dpi=200, facecolor='w')

        corr_heatmap = sns.heatmap(corr_df, annot=True, vmax=1, square=True, cmap=mpt.cm.RdYlBu, fmt='.2g', annot_kws={"fontsize":4})
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=5)

        plt.title('温度湿度相关系数图', fontsize = 8)
        plt.xticks(fontsize=5)
        plt.yticks(fontsize=5)
        # plt.tight_layout()  # 防止x轴标签显示不全
        plt.show()
        return None



if __name__ == "__main__":
    # 读取数据
    aaa = CorrAnaly()
    data_raw1 = pd.read_csv(r'zongbu_2022-09-26_2022-09-28.csv', low_memory=False)
    data_raw2 = pd.read_csv(r'power_2022-09-26_2022-09-28.csv', low_memory=False)
    # 数据预处理
    aaa.data_prepare(data_raw1, data_raw2)
    # 相关性分析
    aaa.data_heatmap()