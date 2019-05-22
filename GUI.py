# encoding=utf-8

"""
GUI 仿真环境3.0
author: 朱林海
"""
import matplotlib.pyplot as plt
from Agent_QL_class import Agent
from Target_class import Target
from GA_class import GA

from scipy import optimize as op
import numpy as np
import time
import sys
import pickle as pk
import random
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

MAZE_H = 600  # grid height
MAZE_W = 900  # grid width

Threshold = 5  # 探测目标被发现了的条件

class GUI(tk.Tk, object):  # (tk.Tk, object)表示Maze类从(tk.Tk, object)两类继承而来
    def __init__(self):
        super(GUI, self).__init__()

        self.target_num = tk.StringVar()  # 绑定目标数量
        self.agent_num = tk.StringVar()  # 绑定探测源agent数量

        self.MGRL_α = tk.StringVar()  # 绑定α参数
        self.MGRL_γ = tk.StringVar()  # 绑定γ参数
        self.MGRL_ε = tk.StringVar()  # 绑定ε参数
        self.MGRL_TrainNum = tk.StringVar()  # 绑定训练次数数据

        self.GA_size = tk.StringVar()  # 绑定种群规模参数
        self.GA_times = tk.StringVar()  # 绑定演进次数参数

        self.title('多目标优化仿真')

        self._build_GUI()

        "存储 画布中的所有直线对象的列表"
        self.list_lines = []

        "存储 探测目标target 探测源agent 绘图圆心坐标信息的数据字典"
        self.dict_target = {"0": None}  # 存储目标的信息，键为标号0~M(字符类型)，值为绘图圆心坐标；标号为0，表示的是空目标，键值为“None”
        self.dict_agent = {}  # 存储探测源的信息，键为标号0~N-1(字符类型)，值为绘图圆心坐标(array数组)

        "存储强化学习时，需要的动作，也即是强化学习的agent动作集"
        self.list_actions = [0]  # 此列表中将存储与探测目标个数相同的若干整数，例如某个整数为i，则其表示的动作含义为“改为探测标号为i的目标”，“0”表示空闲态，不对任何目标探测

        """
        "存储 探测某个目标的探测源标号信息 的数据字典"
        self.dict_target2agent = {"0":0}  #键为目标的标号(标号为0，表示的是空目标)，值为记录与该目标发生探测关系的探测源标号的数量

        "存储 某个探测源所探测的目标的信息 的数据字典"
        self.dict_agent2target = {}  # 键为探测源agent的编号(字符类型)；值为该探测源所探测的目标编号(int)

        """

    def _build_GUI(self):
        ""
        "参数设置区域 frame_setup"
        self.frame_setup = tk.Frame(self , padx=5 , pady=5 , bd=5 , relief="ridge"  )
        self.frame_setup.pack(side="left")
        "仿真环境初始化区域frame_1"
        self.frame_1 = tk.Frame(self.frame_setup ,padx=5 , pady=5 , bd=5 , relief="ridge" )
        self.frame_1.pack()

        self.frame_1_lable_1 = tk.Label(self.frame_1 , text="仿真环境初始化\n").pack(side="top")

        self.frame_1_1 = tk.Frame(self.frame_1)
        self.frame_1_1.pack()
        self.frame_1_1_lable_1 = tk.Label(self.frame_1_1 , text="目标数   :").pack(side="left")
        self.frame_1_1_text_1 = tk.Entry(self.frame_1_1 ,width=10 , textvariable=self.target_num , show=None).pack(side="right")

        self.frame_1_2 = tk.Frame(self.frame_1)
        self.frame_1_2.pack()
        self.frame_1_2_lable_1 = tk.Label(self.frame_1_2, text="探测者数:").pack(side="left")
        self.frame_1_2_text_1 = tk.Entry(self.frame_1_2, width=10, textvariable=self.agent_num ,show=None).pack(side="right")

        self.frame_1_3 = tk.Frame(self.frame_1)
        self.frame_1_3.pack()
        self.frame_1_3_button3 = tk.Button(self.frame_1_3, text="  清空重置 " , command=self.reset_canvas).pack(side="bottom")
        self.frame_1_3_button2 = tk.Button(self.frame_1_3, text="探测者生成" , command=self.produce_agent).pack(side="bottom")
        self.frame_1_3_button1 = tk.Button(self.frame_1_3, text="  目标生成 ", command=self.produce_target).pack(side="bottom")

        "算法初始化区域frame_2"
        self.frame_2 = tk.Frame(self.frame_setup ,padx=5 , pady=5 , bd=5 , relief="ridge" )
        self.frame_2.pack()

        self.frame_2_lable_1 = tk.Label(self.frame_2, text="MARL参数初始化\n").pack(side="top")

        self.frame_2_1 = tk.Frame(self.frame_2)
        self.frame_2_1.pack()
        self.frame_2_1_lable_1 = tk.Label(self.frame_2_1, text="α：          ").pack(side="left")
        self.frame_2_1_text_1 = tk.Entry(self.frame_2_1, width=10, textvariable=self.MGRL_α , show=None).pack(side="right")

        self.frame_2_2 = tk.Frame(self.frame_2)
        self.frame_2_2.pack()
        self.frame_2_2_lable_1 = tk.Label(self.frame_2_2, text="γ：          ").pack(side="left")
        self.frame_2_2_text_1 = tk.Entry(self.frame_2_2, width=10, textvariable=self.MGRL_γ , show=None).pack(side="right")

        self.frame_2_3 = tk.Frame(self.frame_2)
        self.frame_2_3.pack()
        self.frame_2_3_lable_1 = tk.Label(self.frame_2_3, text="ε：          ").pack(side="left")
        self.frame_2_3_text_1 = tk.Entry(self.frame_2_3, width=10, textvariable=self.MGRL_ε , show=None).pack(side="right")

        self.frame_2_4 = tk.Frame(self.frame_2)
        self.frame_2_4.pack()
        self.frame_2_4_lable_1 = tk.Label(self.frame_2_4, text="训练次数：").pack(side="left")
        self.frame_2_4_text_1 = tk.Entry(self.frame_2_4, width=10, textvariable=self.MGRL_TrainNum , show=None).pack(side="right")

        self.frame_2_button1 = tk.Button(self.frame_2 , text="开始执行" , command=self.MARL_run).pack()

        "算法初始化区域frame_3"
        self.frame_3 = tk.Frame(self.frame_setup, padx=5, pady=5, bd=5, relief="ridge")
        self.frame_3.pack()

        self.frame_3_lable_1 = tk.Label(self.frame_3, text="GA参数初始化\n").pack(side="top")

        self.frame_3_1 = tk.Frame(self.frame_3)
        self.frame_3_1.pack()
        self.frame_3_1_lable_1 = tk.Label(self.frame_3_1, text="种群规模：").pack(side="left")
        self.frame_3_1_text_1 = tk.Entry(self.frame_3_1, width=10, textvariable=self.GA_size , show=None).pack(side="right")

        self.frame_3_2 = tk.Frame(self.frame_3)
        self.frame_3_2.pack()
        self.frame_3_2_lable_1 = tk.Label(self.frame_3_2, text="演进次数：").pack(side="left")
        self.frame_3_1_text_2 = tk.Entry(self.frame_3_2, width=10, textvariable=self.GA_times , show=None).pack(side="right")

        self.frame_3_button1 = tk.Button(self.frame_3 , text="开始执行" , command=self.GA_run).pack()


        "算法运行区域 frame_algo_run"
        self.frame_algo_run = tk.Frame(self , padx=5 , pady=5 , bd=5 , relief="ridge"  )
        self.frame_algo_run.pack(side="right")
        "创建并放置画布区域,位于frame_algo_run，用于显示寻路动画"
        self.frame_algo_run_canvas = tk.Canvas(self.frame_algo_run, bg='white',
                           height=MAZE_H ,
                           width=MAZE_W )
        self.frame_algo_run_canvas.pack()


        "创建算法数据输出区域,位于frame_algo_run，用于显示算法运行时的数据"
        self.frame_algo_run_text_1 = tk.Text(self.frame_algo_run , height=10 , width=128 , show=None )
        self.frame_algo_run_text_1.pack()

    "绘制若干探测目标，黄色圆形；同时也负责初始化“self.dict_target”“self.list_actions”的初始化"
    def produce_target(self):
        Number_target = int(self.target_num.get())
        i=1
        while i<Number_target+1:
            origin_x = random.randint(5, 895)
            origin_y = random.randint(5, 595)
            origin = np.array([origin_x, origin_y])  # create origin 圆心，横坐标为0到900 纵坐标为0到600,但为了不画在边缘，所以边界减小5
            oval_target = self.frame_algo_run_canvas.create_oval(
                origin[0] - 5, origin[1] - 5,
                origin[0] + 5, origin[1] + 5,
                fill='yellow')
            "初始化工作"
            self.dict_target[str(i)] = origin  # 赋值为绘图圆心坐标
            self.list_actions.append(i)

            i=i+1
        self.frame_algo_run_text_1.insert("end", "探测目标绘制完成！\n\n")

    "绘制若干探测源(agent)，黑色圆形；同时负责”self.dict_agent“ 的初始化"
    def produce_agent(self):
        Number_agent = int(self.agent_num.get())
        i = 0
        while i < Number_agent:
            origin_x = random.randint(3, 897)
            origin_y = random.randint(3, 597)
            origin = np.array([origin_x, origin_y])  # create origin 圆心，横坐标为0到900 纵坐标为0到600,但为了不画在边缘，所以边界减小1
            oval_agent = self.frame_algo_run_canvas.create_oval(
                origin[0] - 3, origin[1] - 3,
                origin[0] + 3, origin[1] + 3,
                fill='black')
            "初始化工作"
            self.dict_agent[str(i)] = origin

            i = i + 1
        self.frame_algo_run_text_1.insert("end" , "探测源绘制完成！\n\n")

    "清空画布区域，删除已有的探测源、探测者agent信息"
    def reset_canvas(self):
        self.frame_algo_run_canvas.delete("all")

    "刷新画面"
    def render(self):
        time.sleep(0.05)
        self.update()

    "删除所有直线"
    def delet_all_lines(self):
        for line in self.list_lines:
            self.frame_algo_run_canvas.delete(line)
        self.list_lines.clear()

    "删除某个agent的直线"
    def delet_one_line(self , agent):
        if agent.line != None:
            self.frame_algo_run_canvas.delete(agent.line)
        agent.line = None

    "计算回报；绘制连线；在agent探测的当前target.agent_num_list中去掉它；在agent探测的下一target.agent_num_list中加入它"
    "输入参数 agent:一个探测源对象 Target_list:包含所有探测目标对象的列表 MARL:包含所有探测源agent对象的列表"
    def step(self , agent , Target_list , MARL):
        agent_state_now = agent.state_current  # 上一状态状态，即某一探测目标的标号
        agent_state_next = agent.state_next  # 下一状态，即某一探测目标的标号

        if agent_state_next == 0:  # 下一状态为空闲态
            reward = 0  # 回报为0
            #不绘制连线
            agent.line = None

            if agent.lable in Target_list[agent_state_now].agent_num_list:
                Target_list[agent_state_now].agent_num_list.remove(agent.lable)  # 在agent探测的当前target.agent_num_list中去掉它
            Target_list[agent_state_next].agent_num_list.append(agent.lable)  # 在agent探测的下一target.agent_num_list中加入它
            return reward
        else:  # 下一状态不是空闲态
            # 必须先进行：在agent探测的当前target.agent_num_list中去掉它
            if agent.lable in Target_list[agent_state_now].agent_num_list:
                Target_list[agent_state_now].agent_num_list.remove(agent.lable)

            if Target_list[agent_state_next].sum_power(MARL) < Target_list[agent_state_next].power_min:  # 下一状态对应的探测目标未饱和
                # 计算agent与目标的距离
                distance = int( ((agent.position[0] - Target_list[agent_state_next].position[0]) ** 2 + (agent.position[1] - Target_list[agent_state_next].position[1]) ** 2) ** 0.5 )
                #reward = 600-distance  # 回报为100-距离
                reward = 1000 - distance

                # 绘制连线,并保存
                line = self.frame_algo_run_canvas.create_line(Target_list[agent_state_next].position[0] , Target_list[agent_state_next].position[1],
                                                       agent.position[0] , agent.position[1])
                self.list_lines.append(line)
                agent.line = line

                Target_list[agent_state_next].agent_num_list.append(agent.lable)  # 在agent探测的下一target.agent_num_list中加入它

                return reward
            elif Target_list[agent_state_next].sum_power(MARL) >= Target_list[agent_state_next].power_min:  # 下一状态对应的探测目标已饱和
                # 计算agent与目标的距离
                distance = int( ((agent.position[0] - Target_list[agent_state_next].position[0]) ** 2 + (agent.position[1] - Target_list[agent_state_next].position[1]) ** 2) ** 0.5 )
                #reward = -600-distance  # 回报为100-距离
                reward = -1000 - distance

                # 绘制连线,并保存
                line = self.frame_algo_run_canvas.create_line(Target_list[agent_state_next].position[0],Target_list[agent_state_next].position[1],
                                                       agent.position[0], agent.position[1])
                self.list_lines.append(line)
                agent.line = line

                Target_list[agent_state_next].agent_num_list.append(agent.lable)  # 在agent探测的下一target.agent_num_list中加入它

                return reward

    "画图函数，result记录了每次训练优化后的最小功率之和"
    def plot(self, results):
        X = []
        Y = []

        for i in range(len(results)):
            X.append(i)
            Y.append(results[i])

        plt.plot(X, Y)
        plt.xlabel("Times")
        plt.ylabel("Sum_power")
        plt.show()


    "MARL 算法运行控制按钮回调函数"
    def MARL_run(self):
        self.frame_algo_run_text_1.insert("end" , "输入参数已接收，MARL即将开始运行......\n")
        self.frame_algo_run_text_1.insert("end", "MGRL_α:" + self.MGRL_α.get() + "\n")
        self.frame_algo_run_text_1.insert("end", "MGRL_γ:" + self.MGRL_γ.get() + "\n")
        self.frame_algo_run_text_1.insert("end", "MGRL_ε:" + self.MGRL_ε.get() + "\n")
        self.frame_algo_run_text_1.insert("end", "MGRL_TrainNum:" + self.MGRL_TrainNum.get() + "\n\n")

        "初始化算法对象，根据输入的探测源agent数，创建数量相同的agent对象，存储于MARL列表中；根据输入的探测目标target数，创建数量+1(多一个空目标)的target对象，存储于Target_list列表中"
        MARL = []
        for i in range(int(self.agent_num.get())):
            agent = Agent(lable=i , position=self.dict_agent[str(i)] , state_begin=0 , actions=list(range(len(self.list_actions))) , learning_rate=float(self.MGRL_α.get()), reward_decay=float(self.MGRL_γ.get()), e_greedy=float(self.MGRL_ε.get()))
            MARL.append(agent)
        Target_list = []
        for i in range(int(self.target_num.get())+1):
            target = Target(lable=i , position=self.dict_target[str(i)])
            Target_list.append(target)

        "输出一下，看初始化是否正确"
        for i in range(int(self.agent_num.get())):
            print(MARL[i].__dict__)
            print("\n")
        for i in range(int(self.target_num.get())+1):
            print(Target_list[i].__dict__)
            print("\n")

        "开始强化学习训练过程，训练次数由输入的参数控制"
        episode = 0
        result = []  # 记录每次训练的经过优化后的最小功率
        while (episode < int(self.MGRL_TrainNum.get())):  # 根据输入的训练次数进行相应次数的训练，每次训练都对所有agent进行一次学习，即q表更新
            # 每个agent都要进行一次学习
            for i in range(int(self.agent_num.get())):
                print("训练次数：" + str(episode) + "  agent编号：" + str(i))

                # 删除此agent对应的UI界面上的直线
                self.delet_one_line(MARL[i])

                # 刷新UI，留出绘制直线的时间
                self.render()

                # 选择动作
                MARL[i].choose_action(Target_list , MARL , Threshold)

                # 在agent对象内部执行动作，现在对象内部既有当前状态信息，又有下一状态信息
                MARL[i].do_action()

                # 在UI界面执行动作，并获得奖赏。  注：由于此步骤同时涉及探测源 与 探测目标，所以不能写到探测源的抽象类Agent中
                reward = self.step(MARL[i] , Target_list , MARL)

                # agent对象进行一次学习，并在对象内部更新了状态
                MARL[i].learn(reward)

                # 分割输出文本方便阅读
                print("\n")

            """
            一次强化学习训练完成之后，再对分配给每个探测目标Target的探测源agent进行功率优化。并记录功率之和。
            此过程需要Target_list、MARL两个列表中，target和agent的位置信息、agent的分配情况信息
            对每个探测目标target相关联的探测源agent的功率进行线性优化的表达式如下：
            min:sum_power = p1 + p2 + p3 + ...... + pn
                -p1/r1^2 - p2/r2^2 - ...... - pn/rn^2 <= -target.power_min
                0 <= pi <= agent.power_average*2
            """
            power_sum = 0
            power_sum_success = True  # 是否成功优化的标志
            C = 0 #线性规划激励值的权重系数，防止因为数量级的差异覆盖原来Q表中的结果

            # 遍历每个Target，对其所关联的一组agent的功率进行线性优化
            for i in range(int(self.target_num.get()) + 1):
                if i == 0:
                    pass
                else:
                    print("此target编号：" + str(Target_list[i].lable) + " 对其进行探测的agent编号：" + str(
                        Target_list[i].agent_num_list))
                    agent_num = len(Target_list[i].agent_num_list)
                    print("agent_num:" + str(agent_num))

                    rr_list = []  # 记录探测该target的所有agent距其的距离的平方的倒数的相反数
                    for agent_lable in Target_list[i].agent_num_list:
                        rr = (Target_list[i].position[0] - MARL[agent_lable].position[0]) ** 2 + (
                                    Target_list[i].position[1] - MARL[agent_lable].position[1]) ** 2
                        rr_list.append(-1 / rr)
                    print("rr_list:" + str(rr_list))

                    k = 0
                    P = []  # 记录所有agent功率取值范围的元组
                    while k < agent_num:
                        pi = (0, 2 * MARL[0].power_average)  # 功率的取值范围
                        P.append(pi)
                        k = k + 1
                    P = tuple(P)
                    print("P:" + str(P))

                    c = np.array([1] * agent_num)
                    A_ub = np.array([rr_list])
                    B_ub = np.array([-Target_list[i].power_min])
                    print("c:" + str(c))
                    print("A_ub:" + str(A_ub))
                    print("B_ub:" + str(B_ub))
                    res = op.linprog(c=c, A_ub=A_ub, b_ub=B_ub, bounds=P)
                    print("经过优化后的算法参数：" + str(res))

                    if res.success==False:
                        power_sum_success=False
                        break
                    else:
                        # 将优化后的功率记录在每个agent的power_run属性里,并进行线性规划学习，该target的总受到的探测功率的负值为奖励值
                        for power, agent_lable, rr in zip(res.x, Target_list[i].agent_num_list, rr_list):
                            MARL[agent_lable].power_run = power
                            MARL[agent_lable].learn(res.fun*C)
                            print("agent编号：" + str(agent_lable) + "  探测功率：" + str(MARL[agent_lable].power_run) + "  rr：" + str(rr))

                        power_sum = power_sum + res.fun  #记录整体的总功率

            if power_sum_success:
                print("本次训练线性规划成功！！")
                result.append(power_sum)
            else:
                print("本次训练线性规划失败！！")

            """
            # 判断是不是每个探测目标都能够被探测到，若是则结束强化学习
            sign = True
            i=1
            while(i < int(self.target_num.get())+1):
                if Target_list[i].sum_power(MARL) < Target_list[i].power_min:
                    sign = False
                i=i+1
            if sign: break
            """
            episode = episode + 1

        self.plot(result)





    "GA 算法运行控制按钮回调函数"
    def GA_run(self):
        self.frame_algo_run_text_1.insert("end", "输入参数已接收，GA即将开始运行......\n")
        self.frame_algo_run_text_1.insert("end", "GA_size:" + self.GA_size.get() + "\n")
        self.frame_algo_run_text_1.insert("end", "GA_times:" + self.GA_times.get() + "\n\n")

        "初始化算法对象，根据输入的探测源agent数，创建数量相同的agent对象，存储于MARL列表中；根据输入的探测目标target数，创建数量+1(多一个空目标)的target对象，存储于Target_list列表中"
        MARL = []
        for i in range(int(self.agent_num.get())):
            agent = Agent(lable=i, position=self.dict_agent[str(i)], state_begin=0,actions=list(range(len(self.list_actions))))
            MARL.append(agent)
        Target_list = []
        for i in range(int(self.target_num.get()) + 1):
            target = Target(lable=i, position=self.dict_target[str(i)])
            Target_list.append(target)

        population_size = int(self.GA_size.get())
        train_times = int(self.GA_times.get())
        pc = 0.6
        pm = 0.01
        ga = GA(population_size=population_size , pc=pc , pm=pm , MARL=MARL , Target_list=Target_list , train_times=train_times)
        ga.main()


if __name__ == '__main__':
    window = GUI()

    window.mainloop()