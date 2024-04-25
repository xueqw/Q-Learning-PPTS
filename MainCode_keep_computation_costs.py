import random

import numpy as np
import copy


class QL_Heft:
    def __init__(self, task, LearningRate, DiscountFactor, VM):
        self.VM = VM
        self.task = task
        self.LearningRate = LearningRate
        self.DiscountFactor = DiscountFactor

        self.Q_table = np.zeros((task + 1, task + 1), dtype=int)
        for i in range(self.task + 1):
            self.Q_table[0][i] = i
            self.Q_table[i][0] = i

    def read_dag_pre(self):
        self.dag_relation_pre = {}
        filename = 'dag4_PPTS'
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.split()
                task_list_pre = {}
                for line_ in lines:
                    line_list_ = line_.split()
                    if line_list_[0] == line_list[0]:
                        task_list_pre[int(line_list_[1])] = int(line_list_[2])
                        self.dag_relation_pre[int(line_list_[0])] = task_list_pre
        print(self.dag_relation_pre)

    def read_dag_suc(self):
        self.dag_relation_suc = {}
        filename = 'dag4_PPTS'
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.split()
                task_list_suc = {}
                for line_ in lines:
                    line_list_ = line_.split()
                    if line_list_[1] == line_list[1]:
                        task_list_suc[int(line_list_[0])] = int(line_list_[2])
                        self.dag_relation_suc[int(line_list_[1])] = task_list_suc
        print(self.dag_relation_suc)

    def read_pro_avg(self):
        self.computation_costs = []
        self.avg_list = []
        filename = 'computation costs 4_PPTS.txt'
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                temp_sum = 0
                current_line = []
                for i in range(self.VM):
                    current_line.append(int(line.split()[i]))
                    temp_sum += float(line.split()[i])
                self.computation_costs.append(current_line)
                temp_avg = temp_sum / 3
                self.avg_list.append(temp_avg)

    def cul_ranku(self):
        self.read_dag_pre()
        self.read_pro_avg()
        self.ranku_list = {}
        temp_task = self.task
        while temp_task > 0:
            if temp_task == self.task:
                self.ranku_list[temp_task] = self.avg_list[temp_task - 1]
                temp_task -= 1
            else:
                pre_list = list(self.dag_relation_pre[temp_task].keys())
                cost_list = list(self.dag_relation_pre[temp_task].values())
                if len(self.dag_relation_pre[temp_task]) == 1:
                    print(self.ranku_list)
                    self.ranku_list[temp_task] = self.avg_list[temp_task - 1] + cost_list[0] + self.ranku_list[
                        pre_list[0]]
                else:
                    max = 0
                    for suc in pre_list:
                        temp = self.ranku_list[suc] + self.dag_relation_pre[temp_task][suc] + self.avg_list[
                            temp_task - 1]
                        if temp > max:
                            max = temp
                    self.ranku_list[temp_task] = max
                temp_task -= 1
        print(self.ranku_list)

    def get_avaiable(self, ava_set, forbid_set, initial_set):
        ava_set.clear()
        for forbid in initial_set:
            if forbid not in forbid_set:
                ava_set.add(forbid)

    def check_father_node(self, temp_task, forbid_set):
        for father in list(self.dag_relation_suc[temp_task]):
            if father not in forbid_set:
                return False
        return True

    def update_Q(self):
        self.cul_ranku()
        self.read_dag_suc()
        route = []
        entry_task = 1
        for i in range(300000):
            temp_route = [1]
            temp_task = entry_task
            forbid_set = set()
            ava_set = set()
            initial_set = set()
            for j in range(1, self.task + 1):
                initial_set.add(j)
            # Q_table_temp = self.Q_table
            while temp_task != self.task:
                chosen_set = set()
                temp_Q = 0
                if temp_task == 1:
                    forbid_set.add(temp_task)
                    chosen_set = self.dag_relation_pre[temp_task].keys()
                    suc_task = random.choice(list(chosen_set))
                    for choice in chosen_set:
                        if self.Q_table[temp_task][choice] > temp_Q:
                            temp_Q = self.Q_table[temp_task][choice]
                    self.Q_table[temp_task][suc_task] = self.Q_table[temp_task][suc_task] + self.LearningRate * (
                            self.ranku_list[suc_task] + self.DiscountFactor * temp_Q - self.Q_table[temp_task][
                        suc_task])
                    temp_task = suc_task
                    forbid_set.add(temp_task)
                    temp_route.append(temp_task)
                    self.get_avaiable(ava_set, forbid_set, initial_set)
                else:
                    for choice in list(ava_set):
                        if self.check_father_node(choice, forbid_set):
                            chosen_set.add(choice)
                    suc_task = random.choice(list(chosen_set))
                    for choice in chosen_set:
                        if self.Q_table[temp_task][choice] > temp_Q:
                            temp_Q = self.Q_table[temp_task][choice]
                    self.Q_table[temp_task][suc_task] = self.Q_table[temp_task][suc_task] + self.LearningRate * (
                            self.ranku_list[suc_task] + self.DiscountFactor * temp_Q - self.Q_table[temp_task][
                        suc_task])
                    temp_task = suc_task
                    temp_route.append(temp_task)
                    forbid_set.add(temp_task)
                    self.get_avaiable(ava_set, forbid_set, initial_set)

        print(self.Q_table)

    def get_order(self):
        self.update_Q()
        task_order = [1]
        temp_task = 1
        while temp_task != self.task:
            max_q = 0
            temp_task_ = temp_task
            for q in range(len(self.Q_table[temp_task])):
                if self.Q_table[temp_task][q] > max_q and q not in task_order:
                    max_q = self.Q_table[temp_task][q]
                    temp_task_ = q
            task_order.append(temp_task_)
            temp_task = temp_task_
        print(task_order)
        return task_order

    def compute_pcm(self, task_id, processor_id, computation_costs, dag, pcm):
        if task_id not in dag or not dag[task_id]:
            pcm[task_id][processor_id] = computation_costs[task_id - 1][processor_id]
            return pcm[task_id][processor_id]

        if pcm[task_id][processor_id] is not None:
            return pcm[task_id][processor_id]

        max_cost = 0
        for succ_task in dag[task_id]:
            # 仅当任务在不同处理器上时添加通信成本
            min_succ_cost = min(
                self.compute_pcm(succ_task, succ_proc, computation_costs, dag, pcm) +
                computation_costs[succ_task - 1][succ_proc] + computation_costs[task_id - 1][succ_proc] +
                (dag[task_id][succ_task] if succ_proc != processor_id else 0)
                for succ_proc in range(len(computation_costs[0]))
            )
            max_cost = max(max_cost, min_succ_cost)

        pcm[task_id][processor_id] = max_cost
        return pcm[task_id][processor_id]

    def get_pcm_list(self):
        self.read_pro_avg()
        self.read_dag_pre()
        num_processors = len(self.computation_costs[0])
        pcm = {task: [None] * num_processors for task in range(1, self.task + 1)}

        computation_costs = copy.deepcopy(self.computation_costs)
        dag_relation_pre = copy.deepcopy(self.dag_relation_pre)

        # 计算每个任务在每个处理器上的PCM
        for task in range(self.task, 0, -1):  # 从最后一个任务开始向前计算
            for processor in range(num_processors):
                self.compute_pcm(task, processor, computation_costs, dag_relation_pre, pcm)

        # 将PCM转换为列表形式以便处理
        pcm_list: list[list[None]] = [[pcm[task + 1][processor] for processor in range(num_processors)] for task in
                                      range(self.task)]
        return pcm_list

    def select_processor(self):
        pcm_list = self.get_pcm_list()
        task_order = self.get_order()
        EFT_relation = {}
        pre_set = set()
        ava_set = {}
        for current_task in task_order:
            EST = [0, 0, 0]
            if current_task == 1:
                for i in range(self.VM):
                    EST[i] = self.computation_costs[current_task - 1][i]
                LA_eft = EST[0] + pcm_list[0][0]
                min_processor = 0
                for i in range(self.VM):
                    LA_eft_temp = EST[i] + pcm_list[0][i]
                    if LA_eft_temp <= LA_eft:
                        LA_eft = LA_eft_temp
                        EFT_relation[1] = {i + 1: EST[i]}
                        min_processor = i
                        print(EFT_relation)
                ava_set[min_processor + 1] = EST[min_processor]
                pre_set.add(1)
            else:
                for i in range(len(EST)):
                    for pre in list(self.dag_relation_suc[current_task].keys()):
                        if i + 1 == list(EFT_relation[pre].keys())[0]:
                            EST[i] = max(ava_set[i + 1], max(EST[i], list(EFT_relation[pre].values())[0]))
                        else:
                            if i + 1 not in ava_set:
                                EST[i] = max(
                                    (EST[i],
                                     list(EFT_relation[pre].values())[0] + self.dag_relation_suc[current_task][pre]))
                            else:
                                EST[i] = max(ava_set[i + 1], max((EST[i], list(EFT_relation[pre].values())[0] +
                                                                  self.dag_relation_suc[current_task][pre])))
                print(EST)
                EFT = [0, 0, 0]
                for i in range(self.VM):
                    EFT[i] = EST[i] + self.computation_costs[current_task - 1][i]
                LA_eft = EFT[0] + pcm_list[current_task - 1][0]
                min_processor = 0
                for i in range(len(EST)):
                    LA_eft_temp = EST[i] + self.computation_costs[current_task - 1][i] + pcm_list[current_task - 1][i]
                    if LA_eft_temp <= LA_eft:
                        LA_eft = LA_eft_temp
                        EFT_relation[current_task] = {i + 1: EFT[i]}
                        min_processor = i
                ava_set[min_processor + 1] = EFT[min_processor]
                pre_set.add(current_task)

        makespan = max(list(ava_set.values()))
        print("makespan=" + f"{makespan}")


# 创建 QL_Heft 类的实例
ql_heft_instance = QL_Heft(10, 1, 0.8, 3)

ql_heft_instance.select_processor()
