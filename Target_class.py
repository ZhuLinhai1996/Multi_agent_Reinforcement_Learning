# encoding=utf-8

"""
Reinforcement Learning: Q Learning

author: zhulinhai

此文件中包含Target类，是UI界面中一个探测目标target的抽象，并包含了与UI界面有联系的相关属性
"""

class Target(object):
    def __init__(self , lable , position , power_min=100):
        self.lable = lable  # lable是该探测目标target的编号，编号为0，表示这个目标是空目标
        self.position = position  # position是该探测目标target的UI界面上的坐标,是一个二维数组
        self.agent_num_list = []  # agent_num_list是记录探测该目标的探测源agent编号的列表
        self.power_min = power_min  # 此目标能够被准确探测的探测功率最小值，被传入参数初始化

    "计算自身当前的已获得探测功率。MARL为存储所有探测源agent的列表"
    def sum_power(self , MARL):
        sum_pow = 0

        for i in self.agent_num_list:
            sum_pow = sum_pow + MARL[i].power_average/( (MARL[i].position[0]-self.position[0])**2+(MARL[i].position[1]-self.position[1])**2 )
        print("sum_power:"+str(sum_pow))
        return sum_pow