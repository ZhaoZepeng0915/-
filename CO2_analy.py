import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import matplotlib as mpt
from matplotlib import pyplot as plt
import matplotlib.dates as mdate

from IPython.core.interactiveshell import InteractiveShell

plt.rcParams['font.family'] = 'SimHei'  # 防止画图时中文乱码
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
InteractiveShell.ast_node_interactivity = 'all'


class Co2Analy():
    def __init__(self):
        self.df1_prepare = pd.DataFrame()
        self.df2_prepare = pd.DataFrame()

    def data_prepare(self, raw_data1):
        df1 = raw_data1
        '''
            对数据进行预处理，去掉有缺失值的行;创建新的索引
            创建新的两列，分别为日期和时间列
        '''
        df1.dropna(axis=0, how='all', inplace=True)
        df1.fillna(method="ffill", inplace=True)
        df1 = df1.reset_index(drop=True)

        # 生成日期列、时间列
        df1['create_time'] = pd.to_datetime(df1['create_time'], errors='coerce')
        df1['date'] = df1['create_time'].dt.date
        df1['h-m-s'] = df1['create_time'].dt.time

        self.df1_prepare = df1
        return self.df1_prepare

    def co2_analy(self):
        df = self.df1_prepare
        x = df['create_time']
        y1 = df['v2948_K_B_26_1CO2']
        y2 = df['v3443_K_B_26_1新风阀反馈']
        y3 = df['v4235_K_B_26_1回风阀反馈']
        y4 = df['v3047_K_B_26_1变频反馈']


        # 构造多个ax
        fig, ax1 = plt.subplots(figsize=(16, 9))
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax4 = ax1.twinx()

        # 将构造的ax右侧的spine向右偏移
        ax3.spines['right'].set_position(('outward', 60))
        ax4.spines['right'].set_position(('outward', 120))

        # 绘制
        img1, = ax1.plot(x, y1, c='tab:blue')
        img2, = ax2.plot(x, y2, c='tab:orange')
        img3, = ax3.plot(x, y3, c='tab:red')
        img4, = ax4.plot(x, y4, c='tab:green', linestyle='--')

        # 获取对应折线图颜色给到spine ylabel yticks yticklabels
        axs = [ax1, ax2, ax3, ax4]
        imgs = [img1, img2, img3, img4]
        labels = ['1#AHU-回风二氧化碳浓度/ppm', '1#AHU-新风阀开度/%', '1#AHU-回风阀开度/%', '1#AHU-送风机运行频率/Hz']
        for i in range(len(axs)):
            axs[i].spines['right'].set_color(imgs[i].get_color())
            axs[i].set_ylabel(labels[i], c=imgs[i].get_color())
            axs[i].tick_params(axis='y', color=imgs[i].get_color(), labelcolor=imgs[i].get_color())
            axs[i].spines['left'].set_color(img1.get_color())  # 注意ax1是left
        # 设置其他细节
        ax1.set_xlabel('运行时间',size=12)
        # 获得x轴的坐标范围
        # start, end = ax1.get_xlim()
        # # 设置x轴刻度的显示步长
        # plt.xticks(np.linspace(start, end, ))

        ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y:%m:%d:%H:%M:%S'))  # 设置时间标签显示格式
        # plt.xticks(pd.date_range('2022-09-26', '2022-09-28'), rotation=60)

        ax1.set_ylim(0, 1000)
        ax2.set_ylim(0, 100)
        ax3.set_ylim(0, 100)
        ax4.set_ylim(0, 50)
        plt.tight_layout()
        # plt.savefig('n axis.png', dpi=600)
        plt.show()
        print(x)


if __name__ == "__main__":
    # 读取数据
    aaa = Co2Analy()
    data_raw1 = pd.read_csv(r'zongbu_2022-09-26_2022-09-28.csv', low_memory=False)
    # 数据预处理
    aaa.data_prepare(data_raw1)
    # 相关性分析
    aaa.co2_analy()