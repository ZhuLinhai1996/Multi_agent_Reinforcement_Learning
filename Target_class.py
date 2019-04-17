# encoding=utf-8

"""
Reinforcement Learning: Q Learning

author: zhulinhai

此文件中包含Target类，是UI界面中一个探测目标target的抽象，并包含了与UI界面有联系的相关属性
"""

class Target(object):
    def __init__(self , lable , position):
        self.lable = lable  # lable是该探测目标target的编号，编号为0，表示这个目标是空目标
        self.position = position  # position是该探测目标target的UI界面上的坐标,是一个二维数组
        self.agent_num_list = []  # agent_num_list是记录探测该目标的探测源agent编号的列表