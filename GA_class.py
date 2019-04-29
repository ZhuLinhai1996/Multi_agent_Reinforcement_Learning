# encoding=utf-8

"""
author: 朱林海

此文件中包含GA类，是遗传算法的抽象类
"""
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize as op
import sys


class GA(object):
    # 初始化种群 生成chromosome_length大小的population_size个个体的种群
    # chromosome_length是个体二进制编码长度，对应 target数量（包含0，即空目标）
    # agent_num是一个解中维数，对应 agent数目
    # population_size是种群大小

    def __init__(self, population_size , pc, pm , MARL , Target_list , train_times):

        self.population_size = population_size
        self.choromosome_length = len(Target_list)
        self.agent_num = len(MARL)
        self.population=[ [[]] ] # 种群是一个三维矩阵，第一维是agent个体，第二维是所有agent个体组成的若干个解，第三维是population_size个解组成的种群
        self.pc = pc
        self.pm = pm
        self.MARL = MARL  # 存储所有agent的列表
        self.Target_list = Target_list  # 存储所有target的列表
        self.train_times = train_times
        # self.fitness_value=[]
    "种群初始化"
    def species_origin(self):
        for i in range(self.population_size):
            temporary = [[]] # 染色体暂存器

            for k in range(self.agent_num):
                part_temporary = []  # 染色体的一部分，对应某个agent的探测情况

                for j in range(self.choromosome_length):
                    part_temporary.append(random.randint(0,0))  # 产生一个由0组成列表

                xia_biao = random.randint(0, self.choromosome_length-1)
                part_temporary[xia_biao] = 1  #  随机将此列表向量的某一位置1，因为一个agent只探测一个目标
                temporary.append(part_temporary)  # 将染色体的一部分加入到染色体中

            self.population.append(temporary[1:])  # 将染色体添加到种群中

        self.population = self.population[1:]  # 去除声明时加入的一个空[[]]



    # a protein realize its function according its structure
    # 目标函数相当于环境 对染色体进行筛选，这里是经过线性优化之后的功率值之和
    def function(self):
        MARL = self.MARL
        Target_list = self.Target_list

        function1 = []
        temporary = self.population
        for i in range(len(temporary)):
            yi_ge_jie =  temporary[i]  # 一个解，即一个所有agent的分配方案，是[ [010],[100],[001],...... ]这样子
            power_sum = 0  # 记录此解经过线性规划后的最小功率
            power_sum_success = True  # 记录是否成功优化

            for agent_num in range(len(yi_ge_jie)):
                for target_num in range(len(yi_ge_jie[0])):
                    if yi_ge_jie[agent_num][target_num]==1:
                        Target_list[target_num].agent_num_list.append(agent_num)  # 根据编码将探测信息写入到Target_list.agent_num_list，方便下面线性规划

            # 遍历每个Target,做线性规划
            for i in range(self.choromosome_length):
                if i == 0:
                    pass
                else:
                    print("此target编号：" + str(Target_list[i].lable) + " 对其进行探测的agent编号：" + str(Target_list[i].agent_num_list))
                    agent_num = len(Target_list[i].agent_num_list)
                    print("agent_num:" + str(agent_num))

                    rr_list = []  # 记录探测该target的所有agent距其的距离的平方的倒数的相反数
                    for agent_lable in Target_list[i].agent_num_list:
                        rr = (Target_list[i].position[0] - MARL[agent_lable].position[0]) ** 2 + (Target_list[i].position[1] - MARL[agent_lable].position[1]) ** 2
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

                    if res.success==False:  # 不存在可行解
                        power_sum_success = False

                    else:
                        # 将优化后的功率记录在每个agent的power_run属性里
                        for power, agent_lable, rr in zip(res.x, Target_list[i].agent_num_list, rr_list):
                            MARL[agent_lable].power_run = power
                            print("agent编号：" + str(agent_lable) + "  探测功率：" + str(MARL[agent_lable].power_run) + "  rr：" + str(rr))

                        power_sum = power_sum + (res.fun)

            # 对此解，做完线性规划，即得到了此解的最小功率和，然后将其存入function1
            if power_sum_success:
                function1.append(power_sum)
                print("此解线性规划成功！！！\n")
            else:
                function1.append("false")
                print("此解线性规划失败！！！")

            # 清空Target_list中，每个target对象的agent_num_list列表
            # 遍历每个Target,清空agent_num_list列表
            for i in range(self.choromosome_length):
                self.Target_list[i].agent_num_list.clear()

        return function1

    # 定义适应度，对适应度函数计算出的值做处理
    def fitness(self, function1):

        fitness_value = []

        num = len(function1)

        for i in range(num):
            if (function1[i] != "false"):
                temporary = function1[i]
            else:
                temporary = 999999999
            # 如果适应度不存在,则定为极大的一个值

            fitness_value.append(temporary)
        # 将适应度添加到列表中

        return fitness_value

    # 计算适应度和
    def sum(self, fitness_value):
        total = 0

        for i in range(len(fitness_value)):
            total += fitness_value[i]
        return total

    # 计算适应度斐伯纳且列表
    def cumsum(self, fitness1):
        for i in range(len(fitness1) - 2, -1, -1):
            # range(start,stop,[step])
            # 倒计数
            total = 0
            j = 0

            while (j <= i):
                total += fitness1[j]
                j += 1

            fitness1[i] = total
            fitness1[len(fitness1) - 1] = 1

    # 3.选择种群中个体最优的个体，对于此情景是适应度较小的。
    def selection(self, fitness_value):
        new_fitness = []  # 单个公式暂存器

        total_fitness = self.sum(fitness_value)  # 将所有的适应度求和

        for i in range(len(fitness_value)):  # 将所有个体的适应度正则化
            new_fitness.append(fitness_value[i] / total_fitness)

        self.cumsum(new_fitness)

        ms = []  # 存活的种群

        population_length = pop_len = len(self.population)
        # 求出种群长度
        # 根据随机数确定哪几个能存活

        for i in range(pop_len):
            ms.append(random.random())
        # 产生种群个数的随机值
        # ms.sort()
        # 存活的种群排序
        fitin = 0
        newin = 0
        new_population = new_pop = self.population

        # 轮盘赌方式
        while (newin<pop_len)and(fitin<len(new_fitness)):
            if (ms[newin] > new_fitness[fitin]):  # 此处为控制求最大值还是最小值
                new_pop[newin] = self.population[fitin]
                newin += 1
            else:
                fitin += 1

        self.population = new_pop

    # 4.交叉操作
    def crossover(self):
        # pc是概率阈值，选择单点交叉还是多点交叉，生成新的交叉个体，这里没用
        pop_len = len(self.population)

        for i in range(pop_len - 1):

            if (random.random() < self.pc):
                cpoint = random.randint(0, len(self.population[0]))
                # 在种群个数内随机生成单点交叉点
                temporary1 = []
                temporary2 = []

                temporary1.extend(self.population[i][0:cpoint])
                temporary1.extend(self.population[i + 1][cpoint:len(self.population[i])])
                # 将tmporary1作为暂存器，暂时存放第i个染色体中的前0到cpoint个基因，
                # 然后再把第i+1个染色体中的后cpoint到第i个染色体中的基因个数，补充到temporary2后面

                temporary2.extend(self.population[i + 1][0:cpoint])
                temporary2.extend(self.population[i][cpoint:len(self.population[i])])
                # 将tmporary2作为暂存器，暂时存放第i+1个染色体中的前0到cpoint个基因，
                # 然后再把第i个染色体中的后cpoint到第i个染色体中的基因个数，补充到temporary2后面
                self.population[i] = temporary1
                self.population[i + 1] = temporary2
        # 第i个染色体和第i+1个染色体基因重组/交叉完成

    # 突变
    def mutation(self):
        # pm是概率阈值
        px = len(self.population)
        # 求出种群中所有种群/个体的个数
        py = len(self.population[0][0])
        # 染色体/个体基因的个数
        for i in range(px):  # 遍历所有个体
            for j in range(len(self.population[0])):  # 遍历个体中的所有基因，基因是形如[001]的列表
                if (random.random() < self.pm):
                    # 需要先将原值都置为0，再选一个置1，才能保证，只有一个1
                    for k in range(len(self.population[i][j])):
                        self.population[i][j][k]=0

                    mpoint = random.randint(0, py - 1)
                    # 将第mpoint位置进行单点随机变异，变为0或者1
                    self.population[i][j][mpoint] = 1

    # 寻找最好的适应度和个体(本情境下是适应度越低越好)
    def best(self , fitness_value):

        px = len(self.population)
        bestindividual = []  # 记录最好个体，即最优解
        bestfitness = fitness_value[0]  # 记录最好适应度
        # print(fitness_value)

        for i in range(1, px):
            # 循环找出最有的适应度，适应度最小的也就是最好的个体
            if (fitness_value[i] < bestfitness):
                bestfitness = fitness_value[i]
                bestindividual = self.population[i]

        return [bestindividual, bestfitness]

    def plot(self, results):
        X = []
        Y = []

        for i in range(self.train_times):
            X.append(i)
            Y.append(results[i][0])

        plt.plot(X, Y)
        plt.show()

    def main(self):

        results = [[]]
        fitness_value = []
        fitmean = []

        #population = pop = self.species_origin()
        self.species_origin()

        for i in range(self.train_times):
            print("\n\n迭代次数："+str(i))
            function_value = self.function()
            # print('fit funtion_value:',function_value)
            fitness_value = self.fitness(function_value)
            # print('fitness_value:',fitness_value)

            best_individual, best_fitness = self.best(fitness_value)  # 找最优的个体，即解，best_individual；和其对应的适应度best_fitness
            print("最优解："+str(best_individual))
            print("最优解的适应度："+str(best_fitness))
            results.append([best_fitness/10000 , best_individual])  #/10000 是减小尺度方便作图
            # 将最优解和最好的适应度保存
            self.selection(fitness_value)
            self.crossover()
            self.mutation()
        results = results[1:]
        #results.sort()
        self.plot(results)

