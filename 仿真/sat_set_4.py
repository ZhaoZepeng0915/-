'''
   法4 当冷需求被满足时，利用预测模型进行优化
'''
def calculat_tempSP_supplyAir_AHU(damperPosition_VAVbox,  # VAVbox风阀开度
                                  volumeFlowRate_supplyAir_VAVbox,  # VAVbox送风量_体积流量
                                  temp_zone,  # 室内温度
                                  tempSP_zone,  # 室内温度设定值_供冷
                                  COP_chiller,  # 主机COP
                                  elecPower_fan_AHU,  # AHU风机电功率
                                  temp_supplyAir_AHU,  # AHU送风温度
                                  temp_returnAir_AHU,  # AHU回风温度
                                  temp_outdoorAir_AHU,  # AHU新风温度
                                  fraction_outdoorAir_AHU,  # AHU新风比例
                                  load_coolingCoil_AHU):  # 盘管冷负荷

    # 参数设定

    sp_res = -0.3
    sp_res_max = -2.0
    sp_trim = 0.2
    n_zone = 1 #给定VAVbox的数量
    ign = 1 #给定ignore的值
    cop = COP_chiller
    sp_sat = 0.0000000000#########
    damper = [damperPosition_VAVbox]  # 需要提前无量纲化
    temp_k = [temp_zone]
    temp_set_k = [tempSP_zone]
    r_ele = 0.75  # 电价
    alpha = fraction_outdoorAir_AHU
    temp_r = temp_returnAir_AHU
    temp_n = temp_outdoorAir_AHU
    temp_s = temp_supplyAir_AHU
    sum_v_cur, sum_v_alt = 0, 0

    # request计算和mode判断
    req_zone = []
    for i in range(n_zone):
        if ((damper[i] < 0.9) | (temp_k[i] - temp_set_k[i] < 0.5)):
            req_zone[i] = 0
        if ((damper[i] >= 0.9) & (temp_k[i] - temp_set_k[i] >= 0.5) & (temp_k[i] - temp_set_k[i] <= 1)):
            req_zone[i] = 1
        if ((damper[i] >= 0.9) & (temp_k[i] - temp_set_k[i] > 1)):
            req_zone[i] = 2
    req = sum(req_zone)
    mode = 1 if (req > ign) else mode = 0
    # 供冷需求未被满足的情况
    if mode == 1:
        delta_t = (req - ign) * sp_res
        if delta_t > sp_res_max:  # 此处是两个负数
            sp_sat_alt = sp_sat + delta_t
        else:
            sp_sat_alt = sp_sat + sp_res_max

    # 供冷需求已满足的情况（基于成本预测模型）
    if mode == 0:
        vector_sat = []
        del_t = 0.2  # 选择的温度向量中各个温度之间的差值
        for i in range(5):
            vector_sat[i] = sp_sat + (i - 2) * del_t
        # 风机运行成本预测模型
        def cost_fan(temp):
            p_cur = elecPower_fan_AHU,  # AHU当前风机电功率
            v_alt, v_cur = [], []  #每个VAVbox的下一时刻和当前时刻的送风量
            temp_indoor, temp_sp_cur = [], []  # 每个VAVbox的区域温度和室内设定温度
            # 写入列表
            v_cur.append(volumeFlowRate_supplyAir_VAVbox)
            temp_indoor.append(temp_zone)
            temp_sp_cur.append(tempSP_zone)
            temp_s_alt = temp

            for i in n_zone:
                v_alt[i] = v_cur[i] * (temp_indoor[i] - temp_sp_cur[i]) / (temp_indoor[i] - temp_s_alt[i])

            sum_v_alt = sum(v_alt)
            sum_v_cur = sum(v_cur)
            p_alt = p_cur * (sum_v_alt / sum_v_cur) ** 3
            c_fan_alt = r_ele * p_alt
            cost_fan_alt = c_fan_alt
            return cost_fan_alt

        # 供冷成本预测模型
        def cost_cool(temp_s_alt):
            # 新风比
            p_cool_cur = 1.08 * sum_v_cur * (temp_r - temp_s + alpha * (temp_n - temp_r))
            p_cool_alt = 1.08 * sum_v_alt * (temp_r - temp_s_alt + alpha * (temp_n - temp_r))
            cost_cool_cur = r_ele * p_cool_cur / cop if (p_cool_cur > 0) else cost_cool_cur = 0
            cost_cool_alt = r_ele * p_cool_alt / cop if (p_cool_alt > 0) else cost_cool_alt = 0
            return cost_cool_alt

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
        # cost_sum_cur = cost_cool_cur + cost_fan_cur
        # cost_sum_alt = vector_cost[index_min]
        sp_sat_alt = vector_sat[index_min]
        return sp_sat_alt

    # 送风温度给定固定上下限
    if sp_sat_alt > 18.3:
        sp_sat_alt = 18.3
    if sp_sat_alt < 11.6:
        sp_sat_alt = 11.6
    # 返回温度设定值
    tempSP_supplyAir_AHU = sp_sat_alt

    return tempSP_supplyAir_AHU