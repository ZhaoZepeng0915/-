import pandas as pd

'''

    外部输入参数：室内各个子区域温度反馈；室内各个子区域温度设定值；各vav box的开度；当前送风温度设定值sp_sat_cur;当前送风温度sat_cur
                送风机当前功率power_cur,当前回风温度temp_r;当前新风温度temp_n;新风阀和回风阀的开度反馈；
    程序内给定参数：阈值I(ign)；响应参数sp_res；响应参数上限sp_res_max;成本估计所用的固定温差（步长）delta_t；系统子区域的数量n_zone
                电价r_ele;
    输出参数：下一时刻送风温度设定值sp_sat_alt;下一步的运行成本（各项）；当前的运行成本（各项）；
'''
class SatSetting(object):
    def __init__(self, data1):
        self.data = data1
        # 设置需要给定的参数
        self.ign = 3
        self.sp_res = -0.3
        self.sp_res_max = -2.0
        self.delta_t = 0.2
        self.n_zone = 15
        self.req_cool = 0
        self.sp_sat_alt = 0
        self.power_cur = 0
        self.r_ele = 1
        self.temp_r = 0
        self.temp_n = 0
        self.sat_cur = 0
        self.sum_v_alt = 0
        self.sum_v_cur = 0
        self.damper_n = 0
        self.damper_r = 0
        self.cop = 4
        self.cost_sum_cur = 0
        self.cost_fan_cur = 0
        self.cost_cool_cur = 0
        self.cost_sum_alt = 0
        self.cost_fan_alt = 0
        self.cost_cool_alt = 0



    # 读取来自外部的数据
    def data_prepare(self):
        self.damper_n = 0
        self.damper_r = 0


    # 供冷需求R(req)的计算
    def request_cool(self):
        damper = []  #需要提前无量纲化
        temp_k = []
        temp_set_k = []
        req_zone = []
        for i in range(self.n_zone):
            if ((damper[i] < 0.9) | (temp_k[i] - temp_set_k[i] < 0.5)):
                req_zone[i] = 0
            if ((damper[i] >= 0.9) & (temp_k[i] - temp_set_k[i] >= 0.5) & (temp_k[i] - temp_set_k[i] <= 1)):
                req_zone[i] = 1
            if ((damper[i] >= 0.9) & (temp_k[i] - temp_set_k[i] > 1)):
                req_zone[i] = 2
        req = sum(req_zone)
        self.req_cool = req
        # 供冷需求R(req)的计算
        if req > self.ign :
            return 1
        else:
            return 0

    # 供冷需求R>I
    def setpoint_1(self):
        delta_t = (self.request_cool - self.ign) * self.sp_res
        if delta_t > self.sp_res_max:  # 此处是两个负数
            self.sp_sat_alt = self.sp_sat + delta_t
        else:
            self.sp_sat_alt += self.sp_res_max
        return self.sp_sat_alt

    # 供冷需求R<=I
    def setpoint_0(self):
        sp_sat = self.sp_sat_cur
        vector_sat = []
        for i in range(5):
            vector_sat[i] = sp_sat + (i - 2) * self.delta_t

        # 风机运行成本预测模型
        def cost_fan(temp):
            p_cur = self.power_cur
            v_alt , v_cur = [], []
            temp_indoor, temp_s_cur, = [], []
            temp_s_alt = temp
            for i in self.n_zone:
                v_alt[i] = v_cur[i] * (temp_indoor[i] - temp_s_cur[i]) / (temp_indoor[i] - temp_s_alt[i])
            sum_v_alt = sum(v_alt)
            sum_v_cur = sum(v_cur)
            p_alt = p_cur * (sum_v_alt / sum_v_cur)**3
            c_fan_cur = self.r_ele * p_cur
            c_fan_alt = self.r_ele * p_alt
            self.sum_v_alt = sum_v_alt
            self.sum_v_cur = sum_v_cur
            self.cost_fan_cur = c_fan_cur
            self.cost_fan_alt = c_fan_alt
            return self.cost_fan_alt

        # 供冷成本预测模型
        def cost_cool(temp):
            # 修正新风比的系数
            k = 1
            temp_s_alt = temp
            temp_r = self.temp_r
            temp_n = self.temp_n
            alpha = self.damper_n/(self.damper_n + self.damper_r)
            p_cool_cur = 1.08 * self.sum_v_cur * (temp_r - self.sat_cur + k * alpha * (temp_n -temp_r))
            p_cool_alt = 1.08 * self.sum_v_alt * (temp_r - temp_s_alt + k * alpha * (temp_n - temp_r))
            if p_cool_cur > 0:
                self.cost_cool_cur = self.r_ele * p_cool_cur / self.cop
            else:
                self.cost_cool_cur = 0

            if p_cool_alt > 0:
                self.cost_cool_alt = self.r_ele * p_cool_alt / self.cop
            else:
                self.cost_cool_alt = 0
            return self.cost_cool_alt

        # 选择最低成本的送风温度设定值
        vector_cost_fan = []
        vector_cost_cool = []
        vector_cost = []
        for j in vector_sat:
            cost_i = cost_fan(j) + cost_cool(j)
            vector_cost_fan.append(cost_fan(j))
            vector_cost_cool.append(cost_cool(j))
            vector_cost.append(cost_i)

        index_min = vector_cost.index(min(vector_cost))
        self.cost_sum_cur =self.cost_cool_cur + self.cost_fan_cur
        self.cost_sum_alt = vector_cost[index_min]
        self.sp_sat_alt = vector_sat[index_min]
        return self.sp_sat_alt

    # 设定值的输出：
    def data_output(self):
        qqq=1

    # 优化修正系数k
    # 运行数据的可视化


if __name__ == "__main__":
    aaa = SatSetting()
    aaa.data_prepare()
    mode = aaa.request_cool()
    if mode == 1:
        sp_sat = aaa.setpoint_1()
    if mode == 0:
        sp_sat = aaa.setpoint_0()
    aaa.data_output()




