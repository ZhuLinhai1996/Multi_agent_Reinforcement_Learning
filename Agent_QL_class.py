# encoding=utf-8

"""
Reinforcement Learning: Q Learning

author: zhulinhai

此文件中包含Agent类，是Qlearning中一个智能体agent的抽象，并包含了用于强化学习的相关属性与操作，和与UI界面有联系的相关属性
"""

import numpy as np
import pandas as pd

class Agent(object):

    def __init__(self, lable , position , state_begin , actions, learning_rate=0.9, reward_decay=0.9, e_greedy=0.9 ):
        self.lable = lable  # lable是该探测源agent的编号，从0开始
        self.position = position  # position是该探测源agent的UI界面上的坐标,是一个二维数组
        self.line = None  # 此探测源agent对应于UI界面的直线对象，初始值为空

        self.actions = actions  # 动作集
        self.state_current = state_begin  # 迭代学习中，agent的当前状态;并且根据传入的参数设置其初始状态
        self.state_next = None # 迭代学习中，agent的下一状态
        self.action_choose = None  # 迭代学习中的一次学习时，被选中的动作

        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions)

    def choose_action(self):
        self.check_state_exist(self.state_current)

        print("当前所处状态："+ str(self.state_current))

        # action selection
        if np.random.uniform() < self.epsilon: #np.random.uniform(),从一个均匀分布[0,1)中随机采样，返回一个动作值，实现0.9的概率使用过往经验积累

            # choose best action
            state_action1 = self.q_table.loc[self.state_current] #取状态observation对应的一行

            # 获取最大reward值
            #action_value_max = state_action1.values.argmax()
            action_value_max = -100000.000
            for i in state_action1.index:
                if (state_action1[i] > action_value_max):
                    action_value_max = state_action1[i]

            # 对该行数据做乱序处理，防止两个相等的最大值总选择其中某一个
            state_action2 = state_action1.reindex(np.random.permutation(state_action1.index))

            print("当前状态对应的各动作奖赏【已经过乱序处理】："+"\n"+str(state_action2))
            print("各动作奖赏中的最大值："+str(action_value_max))

            #找到最大reward值对应的动作
            #state_action2为'pandas.core.series.Series'类型
            for action in state_action2.index:
                if (state_action2[action] == action_value_max):
                    print("将要采取的动作：" + str(action))
                    #return action
                    self.action_choose = action
                    break  # 由于已经经过了乱序处理，所以找到第一个就跳出

        else:
            # choose random action
            #随机选取动作
            action = np.random.choice(self.actions)
            print("随机选取动作：" + str(action))
            #return action
            self.action_choose = action

    def do_action(self):
        self.state_next = self.action_choose  # 由于q表设计的特殊性，下一状态即为采取的动作

    def learn(self, r):
        self.check_state_exist(self.state_next) #将现状态的下一状态也添加到了q_table中，本人觉得此步不必要，故将其注释
                                                #注释后发现大错特错，因为如果s_没在q_table中，下面程序“q_target = r + self.gamma * self.q_table.ix[s_, :].max()”将出错
        q_predict = self.q_table.ix[self.state_current, self.action_choose]
        '''
        原程序中对下一状态是否为最终状态做了区分处理
        '''
        # if s_ != 'terminal':
        #     q_target = r + self.gamma * self.q_table.ix[s_,:].max() # next state
        # else:
        #     q_target = r # next state is terminal

        '''
        本人使用先暂不区分，日后再深入研究
        '''
        q_target = r + self.gamma * self.q_table.ix[self.state_next, :].max()
        self.q_table.ix[self.state_current, self.action_choose] += self.lr * (q_target - q_predict) # update，更新Q矩阵

        self.state_current = self.state_next  # 更新状态

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