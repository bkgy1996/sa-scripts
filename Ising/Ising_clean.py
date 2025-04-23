import numpy as np
import warnings
import matplotlib.pyplot as plt
import scipy.io as sio
import tkinter as tk
import tkinter.filedialog as FileDialog
import os
import re
import itertools
import xlrd

class Ising:
    def __init__(self, dimension=1, nodes_num=83):
        self.corrs = []
        self.delta_corrs = []
        self.mean_corrs = []
        self.mask = None
        if dimension == 1:
            self.nodes = np.ones((nodes_num, 1))
            # self.nodes[np.where(self.nodes == 0)] = -1
            self.get_J(generate_gauss=True)
            self.h = np.zeros((nodes_num, 1))
            self.E = self.cal_energy(self.nodes, whole_E=True)
            print(self.E)
            self.file_list = []
            self.J_file_list = []
            self.real_data = []
            self.real_nodes = []
            self.data_index = []
            self.P = None
            self.fisher_index = []
            self.states = None
            self.last_real_conf = None
        elif dimension == 2:
            raise NotImplementedError("已弃用二维方法")
            # if nodes_num == 83:
            #     nodes_num = 9
            # self.J = np.ones((nodes_num*nodes_num,nodes_num*nodes_num))
            # self.nodes = np.random.randint(0,2,(nodes_num,nodes_num))
            # self.nodes[np.where(self.nodes==0)] = -1
            # self.E = self.cal_energy(self.nodes)
        else:
            ValueError("初始化函数传入参数维数必须为1或2")

    def clear_nodes(self, dimension=1, whole_E=True):
        if dimension == 1:
            # self.nodes = np.random.randint(0, 2, (self.nodes.shape[0], 1))
            # self.nodes[np.where(self.nodes == 0)] = -1
            self.nodes = np.ones((self.nodes.shape[0], 1))
            self.E = self.cal_energy(self.nodes, whole_E=whole_E)
        elif dimension == 2:
            nodes_num = self.nodes.shape[0]
            self.nodes = np.random.randint(0, 2, (nodes_num, nodes_num))
            self.nodes[np.where(self.nodes == 0)] = -1
            self.E = self.cal_energy(self.nodes)
        else:
            ValueError("清理函数传入参数维数必须为1或2")

    def clear_real_data(self, stream=False):
        self.real_data = []
        self.real_nodes = []
        if stream:
            self.file_list = []
        self.data_index = []
        self.P = None

    def boltzmann(self, kb, T, dE):
        return np.exp(-dE / (kb * T))

    def cal_scheme(self, node_list, method='normal'):
        if not isinstance(node_list, list):
            raise TypeError("cal_scheme应传入时间列表")
        if method == 'normal':
            s = np.mean(node_list, axis=0)
            ss = []
            for t in range(len(node_list)):
                ss.append(np.multiply(node_list[t], node_list[t].T))
            ss = np.mean(ss, axis=0)
            return s, ss
        else:
            tmax = len(node_list)
            js = [self.h + np.dot(self.J, S) for S in node_list]
            # 这里没有排除对角元素，可以通过将连接矩阵对角置零做到
            n_js = np.array(js)
            res_s = 1 / tmax * np.sum(np.tanh(n_js), axis=0)
            # print(res_s.shape)
            sjs = [np.dot(np.tanh(self.h + np.dot(self.J, S)), S.T) for S in node_list]
            res_ss = 1 / tmax * np.sum(sjs, axis=0)
            # print(res_ss.shape)
            return res_s, res_ss

    def real_cal_scheme(self, time_start, time_end, sub_num=0):
        if len(self.real_nodes) < 1:
            raise ValueError("未获得真实节点数据")
        T = self.real_nodes[sub_num].shape[1]
        if isinstance(time_end,str):
            if self.last_real_conf is not None:
                return self.last_real_conf
            time_end = T
        time_interval = time_end - time_start
        if time_end > T or time_interval > T:
            raise ValueError('invalid time')
        nodes = self.real_nodes[sub_num].copy()
        nodes = nodes[:, time_start:time_end]
        m_nodes = np.mean(nodes, axis=1, keepdims=True)
        n = nodes[:, 0]
        n = n[:, np.newaxis]
        sub_bin = np.dot(n, n.T)
        for t in range(1,time_interval):
            n = nodes[:, t]
            n = n[:, np.newaxis]
            sub_bin = sub_bin + np.dot(n, n.T)
        # sub_bin = np.array(sub_bin)
        # print(sub_bin.shape)
        m_SiSj = sub_bin/time_interval
        # print(m_SiSj.shape)
        self.last_real_conf = (m_nodes, m_SiSj)
        return m_nodes, m_SiSj

    def next_hj(self, acc, node_list, time_start, time_end, sub_num=0,sen=None):
        s, ss = self.cal_scheme(node_list)
        # print(ss.shape)
        # print(s.shape)
        #e_s, e_ss = self.real_cal_scheme(time_start, time_end, sub_num=sub_num)
        e_s, e_ss = self.real_cal_scheme(0, 'end', sub_num=sub_num)
        # print(e_s.shape)
        # print(e_ss.shape)
        d_ss = np.diag(np.diag(ss))
        d_e_ss = np.diag(np.diag(e_ss))
        ss = ss - d_ss
        e_ss = e_ss - d_e_ss
        if sen is not None:
            self.h = self.h - acc * (s - e_s) - acc * (s - e_s)*np.diag(np.diag(sen))
            self.J = self.J - acc * (ss - e_ss) - acc * (ss - e_ss)*sen
            # self.h = self.h - acc * (s - e_s) * np.diag(np.diag(sen))
            # self.J = self.J - acc * (ss - e_ss) * sen
        else:
            self.h = self.h - acc * (s - e_s)
            self.J = self.J - acc * (ss - e_ss)
        err = 1
        zero_index = np.where(e_s == 0)
        e_s[zero_index] = 1 / (time_end - time_start)
        err = np.mean(np.abs(e_s - s / e_s))
        return err

    def binary_real_nodes(self, threshold=0.1, write_file=False, prefix='sub'):
        warnings.filterwarnings('error', category=RuntimeWarning)
        np.seterr(all='warn')
        good_index = []
        for i in range(len(self.real_nodes)):
            try:
                self.real_nodes[i][np.where(self.real_nodes[i] >= threshold)] = 1
                self.real_nodes[i][np.where(self.real_nodes[i] < threshold)] = -1
                name = self.file_list[i].split('/')[-1].split('.')[0]
                # with open('good_subject.txt','a') as f:
                #     f.write(name.split('_')[-1]+'.mat\n')
                good_index.append(i)
            except RuntimeWarning as r:
                print(r)
                name = self.file_list[i].split('/')[-1].split('.')[0]
                print('at {here}'.format(here=name))
                with open('brn_runtimewarnings.txt', 'a') as f:
                    f.write(name.split('_')[-1] + '.mat\n')
                continue
        # print(len(good_index))
        self.real_nodes = [self.real_nodes[i] for i in good_index]
        self.file_list = [self.file_list[i] for i in good_index]
        self.data_index = [self.data_index[i] if i < len(self.data_index) else -1 for i in good_index]
        if write_file:
            print("dump file in selected location")
            root = tk.Tk()
            directName = FileDialog.askdirectory(title='please choose directory')
            root.destroy()
            if directName != '':
                for i in range(len(self.real_nodes)):
                    name = prefix + self.file_list[i].split('/')[-1].split('.')[0]
                    path = directName + '/' + name
                    save_mat(+self.real_nodes[i], path)

        warnings.filterwarnings('ignore', category=RuntimeWarning)
        np.seterr(all='ignore')


    def get_real_nodes(self, stream=False, stream_index=-1, directory_name=''):
        if stream:
            self.clear_real_data(stream=True)
            dicName = ''
            if directory_name != '':
                dicName = directory_name
            else:
                print("get file list from directory selected")
                root = tk.Tk()
                dicName = FileDialog.askdirectory(title="get file list from directory selected")
                root.destroy()
                print('you selected {d}'.format(d=dicName))
            files = os.listdir(dicName)
            pattern = re.compile(r'\w+\.mat')
            matNames = []
            for file in files:
                result = re.match(pattern, file)
                if result is not None:
                    matNames.append(result.string)
                    sub_num = file.split('_')[-1]
                    p = re.compile(r'\d+')
                    res = re.findall(p, sub_num)
                    if res is not None:
                        self.data_index.append(int(res[0]))
            for name in matNames:
                self.file_list.append(dicName + '/' + name)

        elif stream_index >= 0:
            N = self.nodes.shape[0]
            data = sio.loadmat(self.file_list[stream_index])
            keys = list(data.keys())
            nodes = data[keys[-1]]
            if nodes.shape[0] == N:
                self.real_nodes.append(nodes)
            elif nodes.shape[1] == N:
                self.real_nodes.append(nodes.T)
            else:
                raise ValueError('real nodes can not match with ising nodes')
        else:
            self.clear_real_data(stream=True)
            dicName = ''
            if directory_name != '':
                dicName = directory_name
            else:
                print("get file list from directory selected")
                root = tk.Tk()
                dicName = FileDialog.askdirectory(title="get file list from directory selected")
                root.destroy()
                print('you selected {d}'.format(d=dicName))
            files = os.listdir(dicName)
            pattern = re.compile(r'\w+\.mat')
            matNames = []
            for file in files:
                result = re.match(pattern, file)
                if result is not None:
                    matNames.append(result.string)
                    sub_num = file.split('_')[-1]
                    p = re.compile(r'\d+')
                    res = re.findall(p, sub_num)
                    if len(res)>0:
                        self.data_index.append(int(res[0]))
            for name in matNames:
                path = dicName + '/' + name
                self.file_list.append(path)
                N = self.nodes.shape[0]
                data = sio.loadmat(path)
                keys = list(data.keys())
                nodes = data[keys[-1]]
                if nodes.shape[0] == N:
                    self.real_nodes.append(nodes)
                elif nodes.shape[1] == N:
                    self.real_nodes.append(nodes.T)
                else:
                    print("bad index {s} found in {a}".format(s=nodes.shape, a=path))
                    raise ValueError('real nodes can not match with ising nodes')

    def normalize_real_nodes(self,section_points=None):
        if len(self.real_nodes) > 0:
            full_nodes = self.real_nodes[0].copy()
            if section_points == "group":
                for i in range(len(self.real_nodes)):
                    full_nodes = np.hstack((full_nodes,self.real_nodes[i]))
            group_m = np.mean(full_nodes, axis=1, keepdims=True)
            group_s = np.std(full_nodes, axis=1, keepdims=True)
            for i in range(len(self.real_nodes)):
                if isinstance(section_points,list):
                    whole = None
                    for j in range(len(section_points)-1):
                        m = np.mean(self.real_nodes[i][:,section_points[j]:section_points[j+1]], axis=1, keepdims=True)
                        s = np.std(self.real_nodes[i][:,section_points[j]:section_points[j+1]], axis=1, keepdims=True)
                        if np.any(s == 0):
                            print(self.data_index[i])
                        part = self.real_nodes[i][:,section_points[j]:section_points[j+1]]
                        part = (part - m)/s
                        if whole is None:
                            whole = part.copy()
                        else:
                            whole = np.hstack((whole,part))
                    self.real_nodes[i] = whole
                elif section_points=='half':
                    whole = None
                    time_length = self.real_nodes[i].shape[1]
                    if time_length%2 != 0:
                        print(self.data_index[i])
                        raise ValueError('half arguments can not be correct')
                    sp = [0,int(time_length/2),time_length]
                    for j in range(len(sp)-1):
                        m = np.mean(self.real_nodes[i][:, sp[j]:sp[j + 1]], axis=1,
                                    keepdims=True)
                        s = np.std(self.real_nodes[i][:, sp[j]:sp[j + 1]], axis=1,
                                   keepdims=True)
                        if np.any(s == 0):
                            print(self.data_index[i])
                        part = self.real_nodes[i][:, sp[j]:sp[j + 1]]
                        part = (part - m) / s
                        if whole is None:
                            whole = part.copy()
                        else:
                            whole = np.hstack((whole, part))
                    self.real_nodes[i] = whole
                elif section_points=='group':
                    self.real_nodes[i] = (self.real_nodes[i] - group_m) / group_s
                else:
                    m = np.mean(self.real_nodes[i], axis=1, keepdims=True)
                    s = np.std(self.real_nodes[i], axis=1, keepdims=True)
                    self.real_nodes[i] = (self.real_nodes[i] - m) / s

    def cal_energy(self, nodes, node_x=-1, node_y=-1, whole_E=False):
        if nodes.shape[1] == 1:
            if node_x != -1:
                length = np.floor(np.sqrt(nodes.shape[0]))
                if length != np.sqrt(nodes.shape[0]):
                    raise ValueError("网络形状不满足二维蒙特卡洛算法")
                i = node_x
                index_y = np.floor(i / length)
                index_x = i - index_y * length
                up = index_y - 1 if index_y > 0 else length - 1
                down = index_y + 1 if index_y < length - 1 else 0
                left = index_x - 1 if index_x > 0 else length - 1
                right = index_x + 1 if index_x < length - 1 else 0

                num_up = int(up * length + index_x)
                num_down = int(down * length + index_y)
                num_left = int(index_y * length + left)
                num_right = int(index_y * length + right)

                part = -self.J[node_x, num_up] * nodes[node_x, 0] * nodes[num_up, 0] \
                       - self.J[node_x, num_down] * nodes[node_x, 0] * nodes[num_down, 0] \
                       - self.J[node_x, num_left] * nodes[node_x, 0] * nodes[num_left, 0] \
                       - self.J[node_x, num_right] * nodes[node_x, 0] * nodes[num_right, 0]
                return part if not whole_E else -self.h[node_x, 0] * nodes[node_x, 0]
            E_h = -np.sum(self.h * nodes)
            # print(self.J.shape)
            E_J = -np.matmul(np.matmul(nodes.T, self.J), nodes) / 2
            # print('E_J is {J},E_h is {h}'.format(J=E_J,h=E_h))
            # print("max is {a},min is {b}".format(a=np.max(self.J),b=np.min(self.J)))
            return E_J if not whole_E else E_h + E_J
            # 相互作用需要编号 分别从左到右从上到下
        else:
            raise NotImplementedError("二维方法已弃用")
            # shape = nodes.shape
            # if node_x != -1 and node_y != -1:
            #     i = node_x
            #     j = node_y
            #     curr = i * shape[0] + j
            #     up = (i - 1) * shape[0] + j if i > 0 else (shape[0] - 1) * shape[0] + j
            #     index_up = i - 1 if i > 0 else shape[0] - 1
            #     down = (i + 1) * shape[0] + j if i < shape[0] - 1 else j
            #     index_down = i + 1 if i < shape[0] - 1 else 0
            #     left = i * shape[0] + j - 1 if j > 0 else i * shape[0] + (shape[1] - 1)
            #     index_left = j - 1 if j > 0 else shape[1] - 1
            #     right = i * shape[0] + j + 1 if j < shape[1] - 1 else i * shape[0]
            #     index_right = j + 1 if j < shape[1] - 1 else 0
            #     part = - self.J[curr, up] * nodes[i, j]*nodes[index_up, j]\
            #            - self.J[curr, left] * nodes[i,j]*nodes[i,index_left]\
            #            - self.J[curr, right] * nodes[i,j]*nodes[i,index_right]\
            #            - self.J[curr, down] * nodes[i,j] * nodes[index_down,j]
            #     return part
            # else:
            #     E = 0
            #     for i in range(shape[0]):
            #         for j in range(shape[1]):
            #             curr = i*shape[0]+j
            #             up = (i-1)*shape[0]+j if i>0 else (shape[0]-1)*shape[0]+j
            #             index_up = i - 1 if i > 0 else shape[0] - 1
            #             down = (i+1)*shape[0]+j if i < shape[0] - 1 else j
            #             index_down = i + 1 if i < shape[0] - 1 else 0
            #             left = i*shape[0]+j - 1 if j >0 else i*shape[0]+(shape[1]-1)
            #             index_left = j - 1 if j> 0 else shape[1] - 1
            #             right = i*shape[0]+j + 1 if j < shape[1] - 1 else i*shape[0]
            #             index_right = j + 1 if j < shape[1] - 1 else 0
            #             part = - self.J[curr, up] * nodes[i, j] * nodes[index_up, j] \
            #                    - self.J[curr, left] * nodes[i, j] * nodes[i, index_left] \
            #                    - self.J[curr, right] * nodes[i, j] * nodes[i, index_right] \
            #                    - self.J[curr, down] * nodes[i, j] * nodes[index_down, j]
            #             E = E - part
            #     return E/2

    def is_neighbor(self, i, j, length):
        # 将始终以第一个维度判断
        index_y = np.floor(i / length)
        index_x = i - index_y * length
        up = index_y - 1 if index_y > 0 else length - 1
        down = index_y + 1 if index_y < length - 1 else 0
        left = index_x - 1 if index_x > 0 else length - 1
        right = index_x + 1 if index_x < length - 1 else 0
        test_y = np.floor(j / length)
        test_x = j - test_y * length
        if (test_x == index_x and test_y == up) or (test_x == index_x and test_y == down):
            return True
        elif (test_x == left and test_y == index_y) or (test_x == right and test_y == index_y):
            return True
        return False

    def get_J(self, generate_ising_2D=False, threshold=0.5, generate_gauss=False, loc=0.0, scale=1.0, stream=False):
        if stream:
            print("try to get corr mats in selected folder")
            root = tk.Tk()
            dicName = FileDialog.askdirectory(title="try to get corr mats in selected folder")
            root.destroy()
            print('you selected {d}'.format(d=dicName))
            files = os.listdir(dicName)
            pattern = re.compile(r'\w+\.mat')
            matNames = []
            for file in files:
                result = re.match(pattern, file)
                if result is not None:
                    matNames.append(result.string)
                    sub_num = file.split('_')[-1]
                    p = re.compile(r'\d+')
                    res = re.findall(p, sub_num)
                    if res is not None:
                        self.data_index.append(int(res[0]))
            for name in matNames:
                self.J_file_list.append(dicName + '/' + name)
            return
        if generate_ising_2D:
            shape = self.J.shape
            if len(shape) != 2 or shape[0] != shape[1]:
                raise IndexError("初始化错误，检查初始化传入参数")
            if np.floor(np.sqrt(shape[0])) != np.sqrt(shape[0]):
                raise IndexError("应传入正当大小一维节点数作为二维类比")
            if self.nodes.shape[1] > 1:
                print("警告:二维Ising模型算法中不需要调用get_J方法")
            self.J = np.zeros((shape[0], shape[1]))
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if self.is_neighbor(i, j, np.sqrt(shape[0])):
                        self.J[i, j] = 1
                    else:
                        self.J[i, j] = 0
            return
        if generate_gauss:
            self.J = np.random.normal(loc=loc, scale=scale, size=(self.nodes.shape[0], self.nodes.shape[0]))
            self.J = self.J - np.diag(np.diag(self.J))
            self.J = np.triu(self.J, 0) + (np.triu(self.J, 0)).T
            return
        shape = self.J.shape
        print("to select connection matrix")
        root = tk.Tk()
        fileName = FileDialog.askopenfilename(filetypes=[('mat file', '*.mat')], title \
            ='choose connection mat file')
        root.destroy()
        data = sio.loadmat(fileName)
        keys = list(data.keys())
        self.J = data[keys[-1]]
        if threshold != -1:
            self.J = np.abs(self.J)
            self.J[np.where(self.J <= threshold)] = 0
            self.J[np.where(self.J == 1)] = 0
            self.J[np.where(self.J > threshold)] = 1
        if len(self.J.shape) != 2:
            raise IndexError("传入的连接结构与当前节点结构不符")
        if shape[0] <= self.J.shape[0] and shape[1] <= self.J.shape[1]:
            self.J = self.J[0:shape[0], 0:shape[1]]
        else:
            raise IndexError("无法将传入连接矩阵与当前网络匹配")

    def evolve_unbalanced(self, T, need_cal_total_E=False, whole_E=False):
        k = 1
        shape = self.nodes.shape
        if shape[1] == 1:
            if need_cal_total_E:
                select_node = np.random.randint(0, shape[0])
                new_nodes = self.nodes.copy()
                new_nodes[select_node, 0] = new_nodes[select_node, 0] * -1
                new_E = self.cal_energy(new_nodes, whole_E=whole_E)
                # print(new_E)
                old_E = self.E
                # print('new_E is {new},old_E is {old}'.format(new=new_E,old=old_E))
                if new_E <= old_E:
                    self.nodes = new_nodes
                    self.E = new_E
                    # print('here')
                else:
                    dE = new_E - old_E
                    # print(dE)
                    B = self.boltzmann(k, T, dE)
                    r = np.random.uniform(0, 1)
                    # print("B is {_b},r is {_r}".format(_b=B,_r=r))
                    if B >= r:
                        self.nodes = new_nodes
                        self.E = new_E
                        # print('there')
            else:
                select_node = np.random.randint(0, shape[0])
                new_nodes = self.nodes.copy()
                new_nodes[select_node, 0] = new_nodes[select_node, 0] * -1
                new_E = self.cal_energy(new_nodes, node_x=select_node, whole_E=whole_E)
                # print(new_E)
                old_E = self.cal_energy(self.nodes, node_x=select_node, whole_E=whole_E)
                if new_E <= old_E:
                    self.nodes = new_nodes
                else:
                    dE = new_E - old_E
                    # print(dE)
                    B = self.boltzmann(k, T, dE)
                    r = np.random.uniform(0, 1)
                    if B >= r:
                        self.nodes = new_nodes
        elif len(shape) == 2:
            raise NotImplementedError
    def evolve_balanced(self, T, need_cal_total_E=False, whole_E=False, no_fluc=False):
        k = 1
        shape = self.nodes.shape
        if shape[1] == 1:
            if need_cal_total_E:
                for i in range(shape[0]):
                    select_node = i
                    new_nodes = self.nodes.copy()
                    new_nodes[select_node, 0] = new_nodes[select_node, 0] * -1
                    new_E = self.cal_energy(new_nodes, whole_E=whole_E)
                    if new_E < self.E:
                        self.nodes = new_nodes
                        self.E = new_E
                        # print('here')
                    elif not no_fluc:
                        dE = new_E - self.E
                        B = self.boltzmann(k, T, dE)
                        r = np.random.uniform(0, 1)
                        if B > r:
                            self.nodes = new_nodes
                            self.E = new_E

            else:
                for i in range(shape[0]):
                    select_node = i
                    new_nodes = self.nodes.copy()
                    new_nodes[select_node, 0] = new_nodes[select_node, 0] * -1
                    new_E = self.cal_energy(new_nodes, node_x=select_node)
                    old_E = self.cal_energy(self.nodes, node_x=select_node)
                    if new_E < self.E:
                        self.nodes = new_nodes
                    else:
                        dE = new_E - old_E
                        B = self.boltzmann(k, T, dE)
                        r = np.random.uniform(0, 1)
                        if B > r:
                            self.nodes = new_nodes
        elif len(shape) == 2:
            raise NotImplementedError


    def get_nodes_energy(self):
        N = self.nodes.shape[0]
        nodes_energy = np.zeros((N,))
        for i in range(N):
            part_0 = self.h[i] * self.nodes[i, 0]
            part_1 = 0
            for j in range(N):
                if j != i:
                    part_1 += self.J[i, j] * self.nodes[i, 0] * self.nodes[j, 0]
            nodes_energy[i] = part_0 + part_1
        return nodes_energy

    def get_fisher_pair(self, num):
        if len(self.fisher_index) == 0:
            N = self.nodes.shape[0]
            index = []
            for i in range(N):
                index.append((i,))
            for i in range(N - 1):
                for j in range(i + 1, N):
                    index.append((i, j))
            self.fisher_index = index
        return self.fisher_index[num]

    def fisher_information(self, nodes):
        if not isinstance(nodes, np.ndarray):
            raise TypeError('fisher_information needs ndarray')
        shape = nodes.shape
        N = shape[1]
        T = shape[0]
        # fim = np.zeros((int((N*N-N)/2+N),int((N*N-N)/2+N)))
        r_nodes = nodes.squeeze()
        index = []
        cast_nodes = []
        for i in range(N):
            index.append((i,))
            cast_nodes.append(r_nodes[:, i])
        for i in range(N - 1):
            for j in range(i + 1, N):
                index.append((i, j))
                cast_nodes.append(r_nodes[:, i] * r_nodes[:, j])
        cast_nodes = np.array(cast_nodes)
        mean_cast = np.mean(cast_nodes, axis=1, keepdims=True)
        second = np.matmul(mean_cast, mean_cast.T)
        if len(self.fisher_index) == 0:
            self.fisher_index = index
        first = None
        for t in range(T):
            # print('processing {here}'.format(here=t))
            c_nodes = []
            for i in range(N):
                c_nodes.append(r_nodes[t, i])
            for i in range(N - 1):
                for j in range(i + 1, N):
                    c_nodes.append(r_nodes[t, i] * r_nodes[t, j])
            c_nodes = np.array(c_nodes)
            c_nodes = c_nodes[:, np.newaxis]
            f = np.matmul(c_nodes, c_nodes.T)
            if first is None:
                first = f
            else:
                first += f
        m_first = first / T
        fim = m_first - second
        # for i in range(fim.shape[0]):
        #     for j in range(i,fim.shape[1]):
        #         left = None
        #         if len(index[i]) == 1:
        #             left = nodes[:,index[i][0],0]
        #         else:
        #             left = nodes[:,index[i][0],0]*nodes[:,index[i][1],0]
        #         right = None
        #         if len(index[j]) == 1:
        #             right = nodes[:,index[j][0],0]
        #         else:
        #             right = nodes[:,index[j][0],0]*nodes[:,index[j][1],0]
        #         first = left*right
        #         second = np.mean(left)*np.mean(right)
        #         m_first = np.mean(first)
        #         fim[i,j] = m_first - second
        #         print('{a} and {b}'.format(a=i,b=j))
        # fim = (np.triu(fim,1)).T + fim
        if fim.shape[0] != int((N * N - N) / 2 + N):
            raise ValueError
        return fim


    def sensitive_matrix(self, vec, back=False):
        if ((not isinstance(vec, np.ndarray)) or len(vec.shape) != 2 or vec.shape[1] != 1) and not back:
            raise TypeError('sensitive_matrix needs one ndarray parameter with shape N*1')
        if ((not isinstance(vec, np.ndarray)) or len(vec.shape) != 2) and back:
            raise TypeError('sensitive_matrix needs one ndarray parameter with shape N*N')
        N = self.nodes.shape[0]
        if not back:
            sen_mat = np.zeros((N, N))
            for i in range(N):
                for j in range(i + 1, N):
                    index = int((2 * N - 3 - i) * (i) / 2 + j - 1) + N
                    sen_mat[i, j] = vec[index, 0]
            sen_mat = sen_mat + sen_mat.T
            d = np.diag(vec[:N, 0])
            sen_mat = sen_mat + d
            return sen_mat
        else:
            length = int((2 * N - 3 - N) * (N) / 2 + N - 1) + N + 1
            vec_mat = np.zeros((length, 1))
            for i in range(N):
                for j in range(i + 1, N):
                    index = int((2 * N - 3 - i) * (i) / 2 + j - 1) + N
                    vec_mat[index, 0] = vec[i, j]
            for i in range(N):
                vec_mat[i, 0] = vec[i, i]
            return vec_mat

    # def AUC(self,real_matrix,model_matrix):
    #     if not isinstance(real_matrix,np.ndarray) or not isinstance(model_matrix,np.ndarray):
    #         raise TypeError('AUC needs 2 ndarray parameters')
    #     if real_matrix.shape != model_matrix.shape:
    #         raise ValueError('AUC shape of 2 matrix should be equal')
    #     threshold = [0.1*i for i in range(1,10)]
    #     index_p = np.where(real_matrix>0)
    #     index_n = np.where(real_matrix==0)



def save_mat(mat, name, data_name='data'):
    if isinstance(name, str) and isinstance(data_name, str):
        sio.savemat(name + ".mat", {data_name: mat})
    else:
        raise TypeError("save_mat 参数类型错误")


def load_mat(key=-1, path=''):
    fileName = path
    if path == '':
        print("get mat from file selected")
        root = tk.Tk()
        fileName = FileDialog.askopenfilename(defaultextension='.mat', filetypes=[('mat file', '.mat')], title \
            ='get mat from file selected')
        root.destroy()
    if fileName != '':
        mat = sio.loadmat(fileName)
        keys = list(mat.keys())
        # print('keys are {ks} you select {k}'.format(ks=keys,k=keys[key]))
        print('you select: ' + fileName)
        return mat[keys[key]]
    else:
        print('can not find the file')
        return None


def load_npy(path=''):
    fileName = path
    if fileName == '':
        root = tk.Tk()
        fileName = FileDialog.askopenfilename(defaultextension='.npy', filetypes=[('npy file', '.npy')], title \
            ='get npy from file selected')
        root.destroy()
    dic = np.load(fileName, allow_pickle=True)
    dic = dic.item()
    if not isinstance(dic, dict):
        raise TypeError('load_npy should return a dict')
    return dic



def extract_info_from_xlsx(type, tag='', path=''):
    if type == 'TAG':
        fileName = path
        if fileName == '':
            print("get xlsx from file selected")
            root = tk.Tk()
            fileName = FileDialog.askopenfilename(defaultextension='.xlsx', filetypes=[('xlsx file', '.xlsx')], title \
                ='get xlsx from file selected')
            root.destroy()
        data = xlrd.open_workbook(fileName)
        table = data.sheet_by_index(0)
        tags = table.row_values(0)
        subjects = table.col_values(0)[1:]
        index = -1
        try:
            index = tags.index(tag)
        except ValueError:
            print('Can not find the tag named Gender')
            return None
        genders = table.col_values(index)[1:]
        info = {}
        for i in range(len(subjects)):
            info[int(subjects[i])] = genders[i]
        return info


def KFold_from_keys(keys:list,fold:int)->list:
    learning_pack = []
    keys_length = len(keys)
    if keys_length//fold < 1:
        raise ValueError('keys is too short for the fold')
    mod = keys_length%fold
    pack_length = np.array([keys_length//fold for i in range(fold)])
    for i in range(mod):
        pack_length[i] += 1
    for i in range(pack_length.shape[0]):
        start_key = np.sum(pack_length[:i])
        end_key = np.sum(pack_length[:i+1])
        test_pack = []
        train_pack = []
        for j in range(keys_length):
            if j >= start_key and j < end_key:
                test_pack.append(keys[j])
            else:
                train_pack.append(keys[j])
        learning_pack.append((train_pack,test_pack))
    return learning_pack

def slice_get_J_worker(sub_num, learning_size, time_slice, time_length, batch_size, err, acc, start_point,
                             real_nodes, Js, hs, corr, plot=False, last_J=None, last_h=None,multiprocess=False):
    nodes_num = real_nodes[0].shape[0]
    worker = Ising(dimension=1, nodes_num=nodes_num)
    worker.get_J(generate_gauss=True, loc=0, scale=0.07)
    if (last_J is not None) and (last_h is not None):
        worker.J = last_J
        worker.h = last_h
    worker.clear_nodes(whole_E=True)
    worker.real_nodes = real_nodes.copy()
    # nodes_list = []
    qs = []
    ms = []
    T = 2
    exam_length = learning_size
    fc = corr - np.diag(np.diag(corr))
    sims = [0]
    if batch_size > 0:
        raise NotImplementedError
    else:
        for i in range(np.abs(batch_size)):
            t = 0
            while (t + time_slice < learning_size):
                nodes_list = []
                for k in range(time_slice):
                    worker.evolve_balanced(T, need_cal_total_E=True, whole_E=True)
                    nodes_list.append(worker.nodes)
                worker.next_hj(acc, nodes_list, t, t + time_slice)
                t += time_slice
            # sim = np.corrcoef(fc.flatten(),worker.J.flatten())[0,1]
            # sims.append(sim)
            # print(sim)
            test_nodes = worker.nodes.copy()
            for k in range(exam_length):
                worker.evolve_balanced(T, need_cal_total_E=True, whole_E=True)
                test_nodes = np.hstack((test_nodes, worker.nodes))
            nodes_std = np.std(test_nodes, axis=1)
            c_nodes = test_nodes.copy()
            for k in range(nodes_std.shape[0]):
                if nodes_std[k] == 0:
                    c_nodes[k, 0] = - c_nodes[k, 0]
            corr = np.corrcoef(c_nodes)
            corr = corr - np.diag(np.diag(corr))
            sim = np.corrcoef(fc.flatten(), corr.flatten())[0, 1]
            print(sim)
            if sim >= np.max(sims):
                Js[sub_num] = worker.J
                hs[sub_num] = worker.h
                # save_mat(worker.J,sub_num + '_' + str(start_point)+'_Js')
                # save_mat(worker.h, sub_num + '_' + str(start_point) + '_hs')
            # Js[sub_num] = worker.J
            # hs[sub_num] = worker.h
            sims.append(sim)
            if not multiprocess:
                if np.max(sims) >= err:
                    break
                elif i == np.abs(batch_size)-1:
                    Js[sub_num+'_bad'] = worker.J
                    hs[sub_num+'_bad'] = worker.h
                    break
            else:
                if np.max(sims) >= err:
                    save_mat(worker.J,sub_num+'_J',data_name=sub_num+'_J')
                    save_mat(worker.h, sub_num+'_h', data_name=sub_num+'_J')
                    save_mat(sims,sub_num + '_sims', data_name=sub_num + '_sims')
                    break
                elif i == np.abs(batch_size)-1:
                    save_mat(worker.J,sub_num+'_J_bad',data_name=sub_num+'_J_bad')
                    save_mat(worker.h, sub_num+'_h_bad', data_name=sub_num+'_J_bad')
                    save_mat(sims, sub_num + '_sims', data_name=sub_num + '_sims')
                    break

    mod_sta = []
    rea_sta = []
    return qs, sims, mod_sta, rea_sta

if __name__ == "__main__":

    #read roi time series ,normalize and binarize
    # nodes_num represents the number of roi
    test = Ising(dimension=1,nodes_num=21)
    # read .mat time series from selected dir
    test.get_real_nodes()
    # zscore each section separately
    test.normalize_real_nodes(section_points='half')
    # binarize loaded zscored time series based on threshold
    test.binary_real_nodes(write_file=True,prefix='glm_',threshold=0.6)

    #data_fitting
    pattern = re.compile(r'glm_\d+\.mat')
    #current_path = os.getcwd()
    current_path = '/your path'
    print('working on '+current_path)
    files = os.listdir(current_path)
    nodes = {}
    fcs = {}
    all_nodes = None
    for file in files:
        res = re.match(pattern,file)
        if res is not None:
            name = res.string
            sub_num = name.split('.')[0]
            mat = load_mat(path=current_path + '/' + name)
            # here we only use on section
            tot_time = int(np.floor(mat.shape[1]/2))
            part_mat = mat[:,tot_time]
            fc = np.corrcoef(part_mat)
            # if the threshold is too high it may have nan value in fc
            if np.any(np.isnan(fc)):
                continue
            # we concentrate subjects in to one giant timeseries
            if all_nodes is None:
                all_nodes = part_mat.copy()
            else:
                all_nodes = np.hstack((all_nodes, part_mat))
            nodes[sub_num] = part_mat
            fcs[sub_num] = fc
    if len(nodes)<1:
        raise FileNotFoundError('No Nodes Found!Job terminated')
    print('{c} subjects included'.format(c=len(nodes)))
    Js = {}
    hs = {}
    final_sims = []
    TIME_SLICE = 60
    TIME_LENGTH = all_nodes.shape[1]
    LEARNING_SIZE = 10000
    BATCH_SIZE = -10000
    ACC = 0.001
    ERR = 0.98
    # this is an example of fitting group data, you can do similar way for each subject
    learning_nodes = [all_nodes]
    start_point = 0
    corr = np.corrcoef(learning_nodes[0])
    if np.any(np.isnan(corr)):
        raise ValueError('unexpected NAN, check the normalization')
    # if
    final_name = 'all'
    a, sims, ms, rs = slice_get_J_worker(final_name, LEARNING_SIZE, TIME_SLICE, TIME_LENGTH, BATCH_SIZE, ERR,
                                               ACC, start_point, learning_nodes, Js, hs, corr)
    # sims record the similarity between corr and simulated fc during fitting
    save_mat(Js['all'],'Js_hcp_glm_6_block_2',data_name='Js_hcp_glm_6_block_2')
    save_mat(hs['all'], 'hs_hcp_glm_6_block_2', data_name='hs_hcp_glm_6_block_2')

    # calculate FIM and decompose it to get eigenvalues and eigenvectors
    Js_wm = load_mat(path='./Js_hcp_glm_6_block_2.mat')
    hs_wm = load_mat(path='./hs_hcp_glm_6_block_2.mat')
    #
    test = Ising(dimension=1,nodes_num=21)
    test.J = Js_wm
    test.h = hs_wm
    # clear_nodes will calculate initial energy based on parameters just provided
    test.clear_nodes(whole_E=True)
    T = 2
    FIMs = []
    fcs = []
    for i in range(1000):
        # balanced version iterate each node but not randomly select one node
        test.evolve_balanced(T,need_cal_total_E=True,whole_E=True)
    for j in range(5):
        nodes_list = []
        for i in range(100000):
            test.evolve_balanced(T,need_cal_total_E=True,whole_E=True)
            nodes_list.append(test.nodes)
        # calculate FIM using N*T time series
        FIM = test.fisher_information(np.array(nodes_list))
        FIMs.append(FIM)
    mean_FIM = np.mean(FIMs,axis=0)
    vals,vecs = np.linalg.eigh(mean_FIM)
    vals = np.real(vals)
    vals = vals[::-1]
    vecs = np.real(vecs)
    vecs = vecs[:,::-1]
    save_mat(vecs,'vecs_hcp_glm_6_block2',data_name='vecs_hcp_glm_6_block2')
    save_mat(vals, 'vals_hcp_glm_6_block2', data_name='vals_hcp_glm_6_block2')


    from scipy.stats import pearsonr
    vecs_glm = load_mat(path='./vecs_hcp_glm_6_all.mat')
    vals_glm = load_mat(path='./vals_hcp_glm_6_all.mat')
    Js_wm = load_npy(path='./hcp_Js_all_glm_6_new.npy')
    hs_wm = load_npy(path='./hcp_hs_all_glm_6_new.npy')
    test = Ising(dimension=1,nodes_num=21)
    # Figure 3c is simply calculate the std of the projections
    vars_fc = []
    vars_ps = []
    for i in range(231):
        var_fc = []
        var_ps = []
        for key, value in Js_wm.items():
            if not fcs.__contains__(key):
                continue
            vec = vecs_glm[:, i][:, np.newaxis]
            p = Js_wm[key] + np.diag(hs_wm[key][:, 0])
            # sensitive_matrix(back=true) transform any matrix with the same shape of parameter p into the same shape of
            # the eigenvector of FIM
            p_vec = test.sensitive_matrix(p, back=True)
            fc_vec = test.sensitive_matrix(fcs[key],back=True)
            var_ps.append(np.matmul(p_vec.T, vec)[0, 0])
            var_fc.append(np.matmul(fc_vec.T, vec)[0, 0])
        vars_ps.append(var_ps)
        vars_fc.append(var_fc)

    # Figure5 the first 9 dimension of J or h belongs to dmn others belongs to wmn
    # demo of calculating figure 5a
    tags_fc = []
    ps = []
    for key, value in Js_wm.items():
        if not fcs.__contains__(key):
            continue
        vec = vecs_glm[:, 0][:, np.newaxis]
        # sensitive_matrix() transform any matrix with the same shape of the FIM's eigenvector into the same shape of
        # p's shape
        sen = test.sensitive_matrix(vec)
        p = Js_wm[key] + np.diag(hs_wm[key][:, 0])
        p_vec = test.sensitive_matrix(p, back=True)
        ps.append(np.matmul(p_vec.T, vec)[0, 0])
        tags_fc.append(np.mean((fcs[key])[0:9,0:9]))
    # weighted FC by times the absolute value of eigenvectors
    # for key, value in Js_wm.items():
    #     if not fcs.__contains__(key):
    #         continue
    #     vec = vecs_glm[:, 0][:, np.newaxis]
    #     sen = test.sensitive_matrix(vec)
    #     p = Js_wm[key] + np.diag(hs_wm[key][:, 0])
    #     p_vec = test.sensitive_matrix(p, back=True)
    #     ps.append(np.matmul(p_vec.T, vec)[0, 0])
    #     tags_fc.append(np.mean((fcs[key]*np.abs(sen))[0:9,0:9]))

    #Figure 3b and Figure S2
    vecs_glm = load_mat(path='./vecs_hcp_glm_6_all.mat')
    Js_wm = load_npy(path='./hcp_Js_all_glm_6_new.npy')
    hs_wm = load_npy(path='./hcp_hs_all_glm_6_new.npy')
    test = Ising(dimension=1,nodes_num=21)
    ps = []
    for key, value in Js_wm.items():
        p = Js_wm[key] + np.diag(hs_wm[key][:, 0])
        p_vec = test.sensitive_matrix(p, back=True)
        ps.append(p_vec[:,0])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=231)
    pca.fit(np.array(ps))
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(pca.components_,vecs_glm.T)

    #per pack
    #all_pack.npy contains the subject label for n fold cv
    #all_pack is structured in (28,10,2,X) shape. 28 represents 28-fold,we only use 10 fold as an example in the paper.
    #10 represts 10 realization ,2 represents testing sets and training sets
    pack = load_npy(path='./all_pack.npy')
    # subjects are fitted individually basing on the labels in all_pack.npy.
    # the key of the dic is the same as the value of pack
    Js = load_npy(path='./pack_Js.npy')
    hs = load_npy(path='./pack_hs.npy')
    vecs_pack = load_npy(path='./pack_vecs.npy')
    vals_pack = load_npy(path='./pack_vals.npy')
    #task performance
    info = extract_info_from_xlsx('TAG', tag='WM_Task_Acc',
                                  path='./unrestricted_xinyangliu_6_12_2018_2_43_32.xlsx')
    etas_list = []
    tags_list = []
    for ke, valu in pack.items():
        if ke != 10:
            # we only use 10-fold as an example in the paper
            continue
        for i in range(10):
            etas_fold = []
            tags_fold = []
            for fold in range(len(pack[ke][i])):
                train_set = pack[ke][i][fold][1]
                test_set = pack[ke][i][fold][0]
                # test = Ising(dimension=1,nodes_num=21)
                key = str(ke) + '_' + str(i) + '_' + str(fold)
                vals = vals_pack[key + '_vals'].copy()
                vecs = vecs_pack[key + '_vecs'].copy()
                vec0 = vecs[:, 0]
                vec1 = vecs[:, 1]
                etas_plus = []
                tags_plus = []
                subjects = test_set
                for k, v in Js_wm.items():
                    if not subjects.__contains__(k):
                        continue
                    sub_num = k.split('_')[-1]
                    if not (info.__contains__(int(sub_num)) and info[int(sub_num)] != ''):
                        continue
                    p = Js_wm[k] + np.diag(hs_wm[k][:, 0])
                    p_vec = test.sensitive_matrix(p, back=True)
                    projection = np.matmul(p_vec.T, vec0[:, np.newaxis])[0, 0]
                    tags_plus.append(info[int(sub_num)])
                    etas_plus.append(projection)
                r_plus, p_plus = pearsonr(etas_plus, tags_plus)
                etas_fold.append(np.array(etas_plus)) if r_plus>0 else etas_fold.append(-np.array(etas_plus))
                tags_fold.append(tags_plus)
            etas_list.append(etas_fold)
            tags_list.append(tags_fold)
    #after get this cell array of etas, we find optimal alpha and compare it with cpm in matlab

    # figure 6e f, sparse matrix
    rs_pack_per = []
    projection_pack_per = []
    tags_pack_per = []
    vec_pack_per = []
    sen_pack_per = []
    pv_pack_per = []
    for per in range(20):
        percentage = (per) * 5
        # percentage = (20-per)*5
        rs_pack = []
        pv_pack = []
        projection_pack = []
        tags_pack = []
        vec_pack = []
        sen_pack = []
        # test = Ising(dimension=1, nodes_num=21)
        for ke, valu in pack.items():
            if ke!=10:
                # we only use 10-fold as an example in the paper
                continue
            rs_ite = []
            pv_list = []
            vec_list = []
            projection_list = []
            tags_list = []
            sen_list = []
            for i in range(10):
                rs_fold = []
                ps_fold = []
                vs = []
                pv_fold = []
                tags_fold = []
                sen_fold = []
                for fold in range(len(pack[ke][i])):
                    train_set = pack[ke][i][fold][1]
                    test_set = pack[ke][i][fold][0]
                    # test = Ising(dimension=1,nodes_num=21)
                    key = str(ke) + '_' + str(i) + '_' + str(fold)
                    vals = vals_pack[key + '_vals'].copy()
                    vecs = vecs_pack[key + '_vecs'].copy()
                    vec0 = vecs[:,0]
                    vec1 = vecs[:,1]
                    # from sloppy to stiff
                    vec0[np.abs(vec0)<np.percentile(np.abs(vec0),percentage)] = 0
                    vec1[np.abs(vec1) < np.percentile(np.abs(vec1), percentage)] = 0
                    # from stiff to sloppy
                    # vec0[np.abs(vec0) > np.percentile(np.abs(vec0), percentage)] = 0
                    # vec1[np.abs(vec1) > np.percentile(np.abs(vec1), percentage)] = 0
                    projections_plus = []
                    tags_plus = []
                    subjects = test_set
                    for k, v in Js_wm.items():
                        if not subjects.__contains__(k):
                            continue
                        sub_num = k.split('_')[-1]
                        if not (info.__contains__(int(sub_num)) and info[int(sub_num)] != ''):
                            continue
                        p = Js_wm[k] + np.diag(hs_wm[k][:, 0])
                        p_vec = test.sensitive_matrix(p, back=True)
                        projection = np.matmul(p_vec.T, vec0[:, np.newaxis])[0, 0] * 0.4761 + \
                                     np.matmul(p_vec.T, vec1[:, np.newaxis])[0, 0] * (1 - 0.4761)
                        tags_plus.append(info[int(sub_num)])
                        projections_plus.append(projection)
                    r_plus, p_plus = pearsonr(projections_plus, tags_plus)

                    projections_minu = []
                    tags_minu = []
                    subjects = test_set
                    for k, v in Js_wm.items():
                        if not subjects.__contains__(k):
                            continue
                        sub_num = k.split('_')[-1]
                        if not (info.__contains__(int(sub_num)) and info[int(sub_num)] != ''):
                            continue
                        p = Js_wm[k] + np.diag(hs_wm[k][:, 0])
                        p_vec = test.sensitive_matrix(p, back=True)
                        projection = np.matmul(p_vec.T, vec0[:, np.newaxis])[0, 0] * 0.4761 - \
                                     np.matmul(p_vec.T, vec1[:, np.newaxis])[0, 0] * (1 - 0.4761)
                        tags_minu.append(info[int(sub_num)])
                        projections_minu.append(projection)
                    r_minu, p_minu = pearsonr(projections_minu, tags_minu)
                    vs.append(vecs[:, 0][:, np.newaxis] * np.sqrt(vals[0, 0]) + vecs[:, 1][:, np.newaxis] * np.sqrt(
                        vals[0, 1])) if np.abs(r_plus) >= np.abs(r_minu) else vs.append(
                        vecs[:, 0][:, np.newaxis] * np.sqrt(vals[0, 0]) - vecs[:, 1][:, np.newaxis] * np.sqrt(vals[0, 1]))
                    rs_fold.append(np.max([np.abs(r_minu), np.abs(r_plus)]))
                    pv_fold.append(p_plus) if np.abs(r_plus) >= np.abs(r_minu) else pv_fold.append(p_minu)
                    ps_fold.append(projections_plus) if np.abs(r_plus) >= np.abs(r_minu) else ps_fold.append(
                        projections_minu)
                    tags_fold.append(tags_plus) if np.abs(r_plus) >= np.abs(r_minu) else tags_fold.append(
                        tags_minu)
                    sen_fold.append(test.sensitive_matrix(vec0[:, np.newaxis]* 0.4761+vec1[:, np.newaxis] * (1 - 0.4761))) if np.abs(r_plus) >= np.abs(r_minu) else sen_fold.append(test.sensitive_matrix(vec0[:, np.newaxis]* 0.4761-vec1[:, np.newaxis] * (1 - 0.4761)))
                rs_ite.append(rs_fold)
                vec_list.append(vs)
                projection_list.append(ps_fold)
                tags_list.append(tags_fold)
                sen_list.append(sen_fold)
                pv_list.append(pv_fold)
            rs_pack.append(rs_ite)
            vec_pack.append(vec_list)
            projection_pack.append(projection_list)
            tags_pack.append(tags_list)
            sen_pack.append(sen_list)
            pv_pack.append(pv_list)
        rs_pack_per.append(rs_pack)
        vec_pack_per.append(vec_pack)
        projection_pack_per.append(projection_pack)
        tags_pack_per.append(tags_pack)
        sen_pack_per.append(sen_pack)
        pv_pack_per.append(pv_pack)

    # if we are using full dataset Figure 7, demo for e-h
    vecs_glm = load_mat(path='./vecs_hcp_glm_6_block2.mat')
    vals_glm = load_mat(path='./vals_hcp_glm_6_block2.mat')
    Js_wm = load_npy(path='./block2_Js.npy')
    hs_wm = load_npy(path='./block2_hs.npy')
    test = Ising(dimension=1, nodes_num=21)
    projections_per = []
    tags_per = []
    sens_per = []
    rs_per = []
    ps_per = []
    for i in range(20):
        per =  i * 5
        tags = []
        projections = []
        vec0 = vecs_glm[:,0].copy()
        vec1 = vecs_glm[:,1].copy()
        vec2 = vecs_glm[:, 2].copy()
        vec0[np.abs(vec0) < np.percentile(np.abs(vec0),per)] = 0
        vec1[np.abs(vec1) < np.percentile(np.abs(vec1), per)] = 0
        vec2[np.abs(vec2) < np.percentile(np.abs(vec2), per)] = 0
        sens_per.append(test.sensitive_matrix(vec0[:, np.newaxis]*0.51 - vec2[:, np.newaxis] * (1 - 0.51)))
        for key, value in Js_wm.items():
            # skip badly fitted subjects
            if len(key.split('_'))>4:
                continue
            sub_num = key.split('_')[1]
            key_J = 'glm' + '_' + sub_num + '_2' + '_J'
            key_h = 'glm' + '_' + sub_num + '_2' + '_h'
            if info.__contains__(int(sub_num)) and info[int(sub_num)] != '':
                p_wm = Js_wm[key_J] + np.diag(hs_wm[key_h][:, 0])
                p_wm_vec = test.sensitive_matrix(p_wm, back=True)
                projection = np.matmul(p_wm_vec.T, vec0[:, np.newaxis])[0, 0] * 0.51 - \
                             np.matmul(p_wm_vec.T, vec2[:, np.newaxis])[0, 0] * (1 - 0.51)
                # remove one extremely bad subject
                if info[int(sub_num)] < 30:
                    continue
                projections.append(projection)
                tags.append(info[int(sub_num)])
        r,p = pearsonr(projections,tags)
        rs_per.append(r)
        ps_per.append(p)

    #Figure S7
    vecs_glm = load_mat(path='./vecs_hcp_glm_6_all.mat')
    vals_glm = load_mat(path='./vals_hcp_glm_6_all.mat')
    Js_wm = load_npy(path='./hcp_Js_all_glm_6_new.npy')
    hs_wm = load_npy(path='./hcp_hs_all_glm_6_new.npy')
    fc_ps_std = load_mat(path='./fcs_ps_std.mat')
    fc_ps_std = np.squeeze(fc_ps_std)
    info = extract_info_from_xlsx('TAG', tag='WM_Task_Acc',
                                  path='./unrestricted_xinyangliu_6_12_2018_2_43_32.xlsx')
    test = Ising(dimension=1, nodes_num=21)
    ps = []
    tags = []
    weigths = np.sqrt(vals_glm[0,:])
    for key,value in Js_wm.items():
        sub_num = key.split('_')[1]
        wm_key = 'glm_'+sub_num
        if not Js_wm.__contains__(wm_key):
            continue
        p_wm = Js_wm[wm_key] + np.diag(hs_wm[wm_key][:, 0])
        p_wm_vec = test.sensitive_matrix(p_wm,back=True)
        if not (info.__contains__(int(sub_num)) and info[int(sub_num)] != ''):
            continue
        ps_list = []
        for i in range(231):
            vec = vecs_glm[:,i][:,np.newaxis]
            ps_list.append(np.matmul(p_wm_vec.T,vec)[0,0])
        ps.append(ps_list)
        tags.append(info[int(sub_num)])
    ps = np.array(ps)
    tags = np.squeeze(np.array(tags))
    import optimization as op
    rs = []
    ws = []
    pvals_list = []
    signs = []
    for i in range(1,200):
        per = i*0.5
        X,y,_w = op.data_prepare(ps,tags,weigths,percentage=per)
        scoreA,scoreB,rvals,pvals = op.sign_weight_cross_val(X,y,n_splits=4,fit_intercept=True)
        rs.append(np.abs(rvals))
        pvals_list.append(pvals)
        ws.append(_w)
        signs.append(scoreB)