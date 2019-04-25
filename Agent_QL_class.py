# encoding=utf-8

"""
Reinforcement Learning: Q Learning

author: zhulinhai

此文件中包含Agent类，是Qlearning中一个智能体agent的抽象，并包含了用于强化学习的相关属性与操作，和与UI界面有联系的相关属性
"""

import numpy as np
import pandas as pd

class Agent(object):

    def __init__(self, lable , position , actions , power_average=100000 , state_begin=0, learning_rate=0.9, reward_decay=0.9, e_greedy=1.0 ):
        self.lable = lable  # lable是该探测源agent的编号，从0开始
        self.position = position  # position是该探测源agent的UI界面上的坐标,是一个二维数组
        self.line = None  # 此探测源agent对应于UI界面的直线对象，初始值为空
        self.power_average = power_average  # 此探测源的探测功率均值,用于任务分配阶段 , 被传入参数初始化
        self.power_run = 0  # 此探测源经过任务分配优化、功率优化之后的最终功率。是最终的优化结果。

        self.actions = actions  # 动作集
        self.state_current = state_begin  # 迭代学习中，agent的当前状态;并且根据传入的参数设置其初始状态
        self.state_next = None # 迭代学习中，agent的下一状态
        self.state_next_tran = None #迭代学习中，寄存每次训练的下一状态信息，因为每次学完后，state_current都会被重置为0，所以需要额外变量存储上一次训练的下一状态
        self.action_choose = None  # 迭代学习中的一次学习时，被选中的动作

        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions)

    "传入参数Target_list为记录所有探测目标对象的列表 , MARL是存储所有探测源agent对象的列表 ,threshold为饱和阈值"
    def choose_action(self , Target_list , MARL , threshold):
        self.check_state_exist(self.state_current)

        print("当前所处状态："+ str(self.state_current))
        print("动作选取前q表：" + "\n" + str(self.q_table))

        # action selection
        if np.random.uniform() < self.epsilon: #np.random.uniform(),从一个均匀分布[0,1)中随机采样，返回一个动作值，实现0.9的概率使用过往经验积累

            # choose best action
            state_action1 = self.q_table.loc[self.state_current] #取状态observation对应的一行

            "动作选择策略需要考虑三方面：Q值 距离 饱和程度，然后取最大值。融合计算方法如下："
            "非0，即非空闲状态：Q值 - 距离 - 饱和程度*100"
            "0，即空闲状态：Q值 - 平均距离(用对角线距离，1080) - 饱和程度(设为0)"
            "按照上面的计算方法，对state_action1进行加工"
            """
            #饱和程度按探测者个数算
            for i in state_action1.index:
                if i==0:
                    state_action1[i] = state_action1[i] - 1080 - 0
                else:
                    if len(Target_list[i].agent_num_list)<threshold:
                        state_action1[i] = state_action1[i] - int( ((self.position[0] - Target_list[i].position[0]) ** 2 + (self.position[1] - Target_list[i].position[1]) ** 2) ** 0.5 )
                    else:
                        state_action1[i] = state_action1[i] - int( ((self.position[0] - Target_list[i].position[0]) ** 2 + (self.position[1] - Target_list[i].position[1]) ** 2) ** 0.5 ) - len(Target_list[i].agent_num_list） * 100 
            """
            #饱和程度按功率算
            for i in state_action1.index:
                if i==0:
                    state_action1[i] = state_action1[i] - 1080 - 0
                else:
                    # 动作价值不考虑饱和界限
                    distance = ((self.position[0] - Target_list[i].position[0]) ** 2 + (self.position[1] - Target_list[i].position[1]) ** 2) ** 0.5
                    if distance>450:  # 当距离超过半图距离时，将距离因素占比急剧增大，实现超出探测距离的效果
                        state_action1[i] = state_action1[i] - int(distance)*1000 - int(Target_list[i].sum_power(MARL)) * 10
                    else:
                        state_action1[i] = state_action1[i] - int(distance) - int(Target_list[i].sum_power(MARL)) * 10

            # 获取最大q值
            #action_value_max = state_action1.values.argmax()
            action_value_max = -10000000.000
            for i in state_action1.index:
                if (state_action1[i] > action_value_max):
                    action_value_max = state_action1[i]

            # 对该行数据做乱序处理，防止两个相等的最大值总选择其中某一个
            state_action2 = state_action1.reindex(np.random.permutation(state_action1.index))

            print("当前状态对应的动作价值【已经过乱序处理】：" + "\n" + str(state_action2))
            print("各动作奖赏中的最大值："+str(action_value_max))

            #找到最大reward值对应的动作
            #state_action2为'pandas.core.series.Series'类型
            for action in state_action2.index:
                if (state_action2[action] == action_value_max):
                    print("将要采取的动作：" + str(action))
                    #return action
                    self.action_choose = action
                    break  # 由于已经经过了乱序处理，所以找到第一个就跳出

            "由于DataFram切片操作会改变原数据，所以需要对上面的切片操作复原"
            """
            # 饱和程度按探测者个数算
            for i in state_action1.index:
                if i==0:
                    state_action1[i] = state_action1[i] + 1080 + 0
                else:
                    if len(Target_list[i].agent_num_list)<threshold:
                        state_action1[i] = state_action1[i] + int( ((self.position[0] - Target_list[i].position[0]) ** 2 + (self.position[1] - Target_list[i].position[1]) ** 2) ** 0.5 )
                    else:
                        state_action1[i] = state_action1[i] + int( ((self.position[0] - Target_list[i].position[0]) ** 2 + (self.position[1] - Target_list[i].position[1]) ** 2) ** 0.5 ) + len(Target_list[i].agent_num_list) * 100 
            """
            # 饱和程度按功率算
            for i in state_action1.index:
                if i==0:
                    state_action1[i] = state_action1[i] + 1080 + 0
                else:
                    # 动作价值不考虑饱和界限
                    if distance>450:  # 当距离超过半图距离时，将距离因素占比急剧增大，实现超出探测距离的效果
                        state_action1[i] = state_action1[i] + int(distance)*1000 + int(Target_list[i].sum_power(MARL)) * 10
                    else:
                        state_action1[i] = state_action1[i] + int(distance) + int(Target_list[i].sum_power(MARL)) * 10

        else:
            # choose random action
            #随机选取动作
            action = np.random.choice(self.actions)
            print("随机选取动作：" + str(action))
            #return action
            self.action_choose = action

        print("动作选取后q表：" + "\n" + str(self.q_table))

    def do_action(self):
        self.state_next = self.action_choose  # 由于q表设计的特殊性，下一状态即为采取的动作
        self.state_next_tran = self.action_choose

    # 学习策略：不对0状态即空闲态进行学习，把他当做一个特殊的对照；其他照常
    def learn(self, r):
        self.check_state_exist(self.state_next) #将现状态的下一状态也添加到了q_table中
        if self.state_next==0:  # 不对空闲态学习
            pass
        else:
            q_predict = self.q_table.ix[self.state_current, self.action_choose]

            q_target = r + self.gamma * self.q_table.ix[self.state_next, :].max()
            self.q_table.ix[self.state_current, self.action_choose] += self.lr * (q_target - q_predict) # update，更新Q矩阵

        self.state_current = self.state_next  # 更新状态
        #self.state_current = 0  #当前状态重新设置为空闲状态，使其永远都在学习处于空闲状态时，应该采取什么动作

        print("本次学习后q表：" + "\n" + str(self.q_table))

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state
                )
            )