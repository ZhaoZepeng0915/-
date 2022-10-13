'''
   法3 Trim and respond logic（只给定固定上下限）
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
    sp_sat = 0.0000000000#########
    damper = [damperPosition_VAVbox]  # 需要提前无量纲化
    temp_k = [temp_zone]
    temp_set_k = [tempSP_zone]

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
    mode = 1 if req > ign else mode = 0
    # 供冷需求未被满足的情况
    if mode == 1:
        delta_t = (req - ign) * sp_res
    if delta_t > sp_res_max:  # 此处是两个负数
        sp_sat_alt = sp_sat + delta_t
    else:
        sp_sat_alt = sp_sat + sp_res_max
    # 供冷需求已满足的情况
    if mode == 0:
        sp_sat_alt = sp_sat + sp_trim

    # 送风温度给定固定上下限
    if sp_sat_alt > 18.3:
        sp_sat_alt = 18.3
    if sp_sat_alt < 11.6:
        sp_sat_alt = 11.6
    # 返回温度设定值
    tempSP_supplyAir_AHU = sp_sat_alt

    return tempSP_supplyAir_AHU