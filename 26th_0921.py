#!/usr/bin/env python
# coding: utf-8
# # 导入需要的库

# In[1]:

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from IPython.core.interactiveshell import InteractiveShell

plt.rcParams['font.family'] = 'SimHei'  # 防止画图时中文乱码
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
InteractiveShell.ast_node_interactivity = 'all'


# In[2]:


class DataAnalysis(object):
    def __init__(self, data1):
        self.data = data1
        self.df_prepare = pd.DataFrame()

    # 处理异常值
    def data_prepare(self):
        df = self.data
        '''
            对数据进行预处理，去掉有缺失值的行;创建新的索引
            创建新的两列，分别为日期和时间列
        '''
        df.dropna(axis=0, how='all', inplace=True)
        df = df.reset_index(drop=True)
        df['create_time'] = pd.to_datetime(df['create_time'], errors='coerce')
        # 生成日期列、时间列
        df['date'] = df['create_time'].dt.date
        df['h-m-s'] = df['create_time'].dt.time
        df.fillna(method="ffill", inplace=True)
        self.df_prepare = df
        return self.df_prepare


    # 逐时温度变化情况(送风温度及设定值)
    def sat_show(self):
        df = self.df_prepare
        df['index_time'] = df['create_time'].astype(str)
        df.set_index(['index_time'], inplace=True)

        sat_1 = df.loc[:,'v2651_K_B_26_1送风温度']
        sat_2 = df.loc[:,'v2652_K_B_26_2送风温度']
        sat_sp1 = df.loc[:,'v2750_K_B_26_1送风设定']
        sat_sp2 = df.loc[:,'v2751_K_B_26_2送风设定']
        plt.figure(dpi=200, figsize=(10, 5))

        # AHU1送风温度图
        plt.subplot(2, 1, 1)
        plt.plot(sat_1[:4320], 'g', label='sat_1', linewidth=0.8)
        plt.plot(sat_sp1[:4320], 'r', label='sat_sp1', linewidth=0.5, linestyle='--')
        plt.legend(fontsize=6)
        plt.xlabel('Time', fontsize=7)
        plt.xticks(range(0, 4320, 180), rotation=30)
        plt.ylabel('Temperature/℃', fontsize=7)
        plt.title('SAT of AHU-1', fontsize=8)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.7)
        plt.tick_params(labelsize=5)
        # AHU2送风温度图
        plt.subplot(2, 1, 2)
        plt.plot(sat_2[:4320], 'g', label='sat_2', linewidth=0.8)
        plt.plot(sat_sp2[:4320], 'r', label='sat_sp2', linewidth=0.5, linestyle='--')
        plt.legend(fontsize=6)
        plt.xlabel('Time', fontsize=7)
        plt.xticks(range(0, 4320, 180), rotation=30)
        plt.ylabel('Temperature/℃', fontsize=7)
        plt.title('SAT of AHU-2', fontsize=8)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.7)
        plt.tick_params(labelsize=5)
        return None

    def freq_show(self):
        df = self.df_prepare
        df['index_time'] = df['create_time'].astype(str)
        df.set_index(['index_time'], inplace=True)

        freq_1 = df.loc[:,'v3047_K_B_26_1变频反馈']
        freq_2 = df.loc[:,'v3048_K_B_26_2变频反馈']
        freq_sp1 = df.loc[:,'v3245_K_B_26_1变频控制']
        freq_sp2 = df.loc[:,'v3246_K_B_26_2变频控制']
        plt.figure(dpi=200, figsize=(10, 5))

        # AHU1送风频率图
        plt.subplot(2, 1, 1)
        plt.plot(freq_1[:4320], 'g', label='sat_1', linewidth=0.8)
        plt.plot(freq_sp1[:4320], 'r', label='sat_sp1', linewidth=0.5, linestyle='--')
        plt.legend(fontsize=6)
        plt.xlabel('Time', fontsize=7)
        plt.xticks(range(0, 4320, 180), rotation=30)
        plt.ylabel('Freq/Hz', fontsize=7)
        plt.title('Freq of AHU-1', fontsize=8)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.7)
        plt.tick_params(labelsize=5)
        # AHU2送风频率图
        plt.subplot(2, 1, 2)
        plt.plot(freq_2[:4320], 'g', label='sat_2', linewidth=0.8)
        plt.plot(freq_sp2[:4320], 'r', label='sat_sp2', linewidth=0.5, linestyle='--')
        plt.legend(fontsize=6)
        plt.xlabel('Time', fontsize=7)
        plt.xticks(range(0, 4320, 180), rotation=30)
        plt.ylabel('Freq/Hz', fontsize=7)
        plt.title('Freq of AHU-2', fontsize=8)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.7)
        plt.tick_params(labelsize=5)
        return None

    def indoor_temp_show1(self):
        df = self.df_prepare
        df['index_time'] = df['create_time'].astype(str)
        df.set_index(['index_time'], inplace=True)

        temp_2 = df.loc[:,'v0068_VAV72602房间温度']
        temp_3 = df.loc[:,'v0069_VAV72603房间温度']
        temp_4 = df.loc[:, 'v0070_VAV72604房间温度']
        temp_5 = df.loc[:, 'v0071_VAV72605房间温度']
        temp_sp2 = df.loc[:,'v4844_VAV72602设定温度']
        temp_sp3 = df.loc[:,'v4845_VAV72603设定温度']
        temp_sp4 = df.loc[:,'v4846_VAV72604设定温度']
        temp_sp5 = df.loc[:,'v4847_VAV72605设定温度']
        plt.figure(dpi=200, figsize=(10, 5))

        plt.subplot(1, 1, 1)
        plt.plot(temp_2[:4320], 'g', label='temp_2', linewidth=0.8)
        plt.plot(temp_sp2[:4320], 'g', label='temp_sp2', linewidth=0.5, linestyle='--',alpha=0.5)
        plt.plot(temp_3[:4320], 'b', label='temp_3', linewidth=0.8)
        plt.plot(temp_sp3[:4320], 'b', label='temp_sp3', linewidth=0.5, linestyle='--',alpha=0.5)
        plt.plot(temp_4[:4320], 'r', label='temp_4', linewidth=0.8)
        plt.plot(temp_sp4[:4320], 'r', label='temp_sp4', linewidth=0.5, linestyle='--',alpha=0.5)
        plt.plot(temp_5[:4320], color='#CC99FF', label='temp_5', linewidth=0.8)
        plt.plot(temp_sp5[:4320], color='#CC99FF', label='temp_sp5', linewidth=0.5, linestyle='--',alpha=0.5)
        plt.legend(fontsize=6)
        plt.xlabel('Time', fontsize=7)
        plt.xticks(range(0, 4320, 180), rotation=30)
        plt.ylabel('Temp/℃', fontsize=7)
        plt.title('室内温度', fontsize=8)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.7)
        plt.tick_params(labelsize=5)
        return None

    def indoor_temp_show2(self):
        df = self.df_prepare
        df['index_time'] = df['create_time'].astype(str)
        df.set_index(['index_time'], inplace=True)

        temp_2 = df.loc[:, 'v7076_VAV72608房间温度']
        temp_3 = df.loc[:, 'v7078_VAV72610房间温度']
        temp_4 = df.loc[:, 'v7080_VAV72612房间温度']
        temp_5 = df.loc[:, 'v7081_VAV72613房间温度']
        temp_sp2 = df.loc[:,'v4850_VAV72608设定温度']
        temp_sp3 = df.loc[:,'v4852_VAV72610设定温度']
        temp_sp4 = df.loc[:,'v4854_VAV72612设定温度']
        temp_sp5 = df.loc[:,'v4855_VAV72613设定温度']
        plt.figure(dpi=200, figsize=(10, 5))

        plt.subplot(1, 1, 1)
        plt.plot(temp_2[:4320], 'g', label='temp_8', linewidth=0.8)
        plt.plot(temp_sp2[:4320], 'g', label='temp_sp8', linewidth=0.5, linestyle='--',alpha=0.5)
        plt.plot(temp_3[:4320], 'b', label='temp_10', linewidth=0.8)
        plt.plot(temp_sp3[:4320], 'b', label='temp_sp10', linewidth=0.5, linestyle='--',alpha=0.5)
        plt.plot(temp_4[:4320], 'r', label='temp_12', linewidth=0.8)
        plt.plot(temp_sp4[:4320], 'r', label='temp_sp12', linewidth=0.5, linestyle='--',alpha=0.5)
        plt.plot(temp_5[:4320], color='#CC99FF', label='temp_13', linewidth=0.8)
        plt.plot(temp_sp5[:4320], color='#CC99FF', label='temp_sp13', linewidth=0.5, linestyle='--',alpha=0.5)
        plt.legend(fontsize=6)
        plt.xlabel('Time', fontsize=7)
        plt.xticks(range(0, 4320, 180), rotation=30)
        plt.ylabel('Temp/℃', fontsize=7)
        plt.title('室内温度', fontsize=8)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.7)
        plt.tick_params(labelsize=5)
        plt.show()
        return None



# In[3]:

if __name__ == "__main__":
    data = pd.read_csv(r'zongbu3_2022-09-14_2022-09-20.csv',low_memory=False)
    aaa = DataAnalysis(data)
    df_prepare = aaa.data_prepare()
    aaa.sat_show()
    aaa.freq_show()
    aaa.indoor_temp_show1()
    aaa.indoor_temp_show2()


