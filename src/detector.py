"""
2022/2/2
1. read matrix data from file as contact map
2. leverage the contact map to obtain tables needed
3. calculate infomap entropy
2022/02/15
1. delete unnecessary for loop
2. change the dp_table from np array into map
"""
from __future__ import division

import pdb

import numpy as np
import math
import copy
from pandas.core.frame import DataFrame

import src.multiTree as MT
import src.TreeNode
import src.compare_performance as comp
import sys
sys.path.append("..")
import KR_norm_juicer as KRnorm

class TADDetector:
    # matrix = None
    # _N_ = None              # the number of bins
    # v_G = None              # the sum of volume
    # k_max = None            # the number of leaves
    # dp_table = None          # the dp table for obtaining the values of the entropy
    # boundary = ([], [])     # the boundary of TADs, the first is for the left, the second is for the right
    # edge_count = None       # the table to keep values of g and v for all the bins
    #                                                                 # the upper triangle for intra, low triangle for inter (g). v = 2 * intra +inter
    # tree_list = []          # to keep all the possible trees which should be deleted after backtracing
    # step_size = None         # estimation for gamma
    #
    # bound_list = []

    def __init__(self, input_matrix, k_scal=1, step_scal=1, KR_norm=False,sim_mode=False,random_float=False):
        self.org_N_ = input_matrix.shape[0]
        if KR_norm==False:
            if sim_mode==True and random_float==False:
                self.matrix = input_matrix.astype(int)
            else:
                self.matrix = input_matrix
        else:
            self.matrix,_ = KRnorm.KRnorm_sym(input_matrix)
        self.process_diag(state=True)
        self.non_zero_pos = None
        self.process_with_zero_rows(state=1)
        self.process_with0(state=0)   # input: Hic matrix
        self.process_withnan(state=1)
        self._N_ = len(self.matrix)   # the number of bins
        self.k_max = int(self._N_/k_scal + 0.5)                # the number of leaves
        # self.edge_count = np.zeros(shape=(self.n + 1, self.n + 1, 2))
        self.edge_count = np.zeros(shape=(self._N_, self._N_))  # the table to keep values of g and v for all the bins
                                                                    # the upper triangle for intra, the lower triangle for inter (g). v = 2*intra +inter
        self.step_scal = step_scal
        self.boundary = ([], [])     # the boundary of TADs, the first is for the left, the second is for the right
        self.input_process()
        # self.tree = mT.MultiTree(self._N_)
        self.bound_list = []
        # self.l_max = math.ceil(self.v_G/self.step_size)
        # # previous step size for acuracy
        if sim_mode==True:
            self.step_size = 1
            self.l_max = math.ceil(self.v_G/self.step_size)
        else:
            # newly set step size
            # the number of steps
            self.l_max = math.ceil(self._N_ * self.step_scal)
            self.step_size = math.floor(self.v_G / self.l_max)
        # print("ac_step_size: ",self.step_size)
        # if self.step_size == 0:  # the value of step size too small
        #     self.step_size = 1
        #     self.l_max = math.ceil(self.v_G/self.step_size)

        self.dp_table = np.zeros(shape=(self._N_, self.l_max+1, self.k_max))
        # self.dp_table = []
        self.table_i = np.zeros(shape=(self._N_, self.l_max+1, self.k_max), dtype=int)
        self.table_t = np.zeros(shape=(self._N_, self.l_max+1, self.k_max), dtype=int)
        self.dp_table[:, :, :] = float("inf")
        self.table_t[:, :, :] = self.l_max + 1
        self.table_i[:, :, :] = self._N_ + 1
        self.min_entro = float("inf")
        self.opt_bound_list = None

    def process_withnan(self, state=0):
        if state == 0:
            return 1
        else:
            self.matrix[np.isnan(self.matrix)] = 0
        return 1

    def process_diag(self, state=True):
        if state:
            row, col = np.diag_indices_from(self.matrix)
            self.matrix[row, col] = 0

    def process_with0(self, state=0):
        if state != 0:
            # find all the zeros
            index = np.where(self.matrix == 0)
            self.matrix[index] = 1e-7
            # pdb.set_trace()

    def process_with_zero_rows(self, state=0):
        if state != 0:
            all_one_vec = np.ones((self.matrix.shape[0], 1))
            non_zero_pos = np.where(np.dot(self.matrix, all_one_vec) != 0)[0]
            if len(np.where(np.dot(self.matrix, all_one_vec) == 0)[0]) == 0:
                return 1
            input_matrix_mod = self.matrix[non_zero_pos]
            self.matrix = input_matrix_mod[:, non_zero_pos]
            self.non_zero_pos = non_zero_pos

    def after_construction_redirecting(self,opt_bound_list):
        for i in range(len(opt_bound_list)):
            bound = opt_bound_list[i]
            # produce the label
            if len(bound[0]) == 1:
                n_samples = bound[1][0] + 1
            else:
                n_samples = bound[1][-1] + 1
            # construct the label vector
            labels = np.ones(shape=(n_samples,))
            if len(bound[0]) == 1:
                labels[:] = 1
            else:
                for j in range(len(bound[1])):
                    start = bound[0][j]
                    end = bound[1][j]
                    labels[start:end + 1, ] = j + 1
            # map to the org indices
            label_redirected = np.zeros(shape=(self.org_N_,))
            label_redirected[self.non_zero_pos] = labels
            new_bound = ([], [])
            for j in range(1, len(bound[1]) + 1):
                pos = np.where(label_redirected == j)
                if len(pos[0]) != 0:
                    left_b = pos[0][0]
                    right_b = pos[0][-1]
                    new_bound[0].append(left_b)
                    new_bound[1].append(right_b)
            opt_bound_list[i] = new_bound
        return opt_bound_list

    # process the input matrix and obtain the self.edge_count table variable for keeping all the intra and inter
    def input_process(self):
        # in_length is the distance between bins
        for in_length in range(1, self._N_):
            # obtain the table along the diagonal of the input matrix
            for i in range(self._N_-in_length):
                j = i + in_length
                intra = self.matrix[i][j]
                if j > 1:
                    intra = intra + self.edge_count[i][j-1]
                if i < self._N_ - 1:
                    intra = intra + self.edge_count[i+1][j]
                if j > 1 and i <self._N_ - 1:
                    intra = intra - self.edge_count[i+1][j-1]
                # put the intra into the edge_count
                if intra > 0:
                    self.edge_count[i][j] = intra
                else:
                    self.edge_count[i][j] = 0
        # calculate the inter
        for i in range(self._N_):
            for j in range(i, self._N_):
                inter = self.edge_count[0][j] + self.edge_count[i][self._N_-1] - 2 * self.edge_count[i][j]
                if i > 1:
                    inter = inter - self.edge_count[0][i-1]
                if j < self._N_ - 1:
                    inter = inter - self.edge_count[j+1][self._N_ - 1]
                if inter < 0:
                    self.edge_count[j][i] = 0
                else:
                    self.edge_count[j][i] = inter

        self.v_G = 2 * self.edge_count[0][self._N_ - 1]

    # calculate the value of g
    # start is the index of the start bin, end is the index of the end bin
    # g is equal to inter
    def cal_g(self, start, end):
        if start > end:
            print("in cal_g, Calculating the value of g, start > end!")
            return 0
        return self.edge_count[end][start]

    # calculate the value of v
    def cal_v(self, start, end):
        if start > end:
            print("in cal_v, Calculating the value of g, start > end!")
            return 0
        elif start < end:
            return 2 * self.edge_count[start][end] + self.edge_count[end][start]
        elif start == end:
            return self.edge_count[start][end]

    # k_set is the number of leaves and can be input when running
    # By default we build a binary tree
    # def dp_process(self, dary=2):
    #     if dary == 2:               # binary
    #         self.dp_process_2d()
    #
    # # for binary tree
    # def dp_process_2d(self):

    def cal_delta(self, l_b, r_b):
        g = self.cal_g(l_b, r_b)
        v = self.cal_v(l_b, r_b)
        if g != 0:
            g_log_g = g * math.log(g, 2)
        else:
            g_log_g = 0
        # if v+g == 0:
        #     pdb.set_trace()
        if v + g == 0:
            v_p_g_log = 0
        else:
            v_p_g_log = (v + g) * math.log((v + g), 2)
        delta = - 2 * g_log_g + v_p_g_log
        delta = delta / self.v_G
        return delta

    def cal_leaf_sum(self, l_b, r_b):
        entropy = 0
        for i in range(l_b, r_b+1):
            li = self.cal_v(i, i)
            if li == 0:
                continue
            else:
                entropy = entropy + (li * math.log(li, 2))
        entropy = entropy / self.v_G
        return entropy

    def cal_leaf_entro(self, l_b, r_b, gi, vi):
        entropy = 0
        for i in range(l_b, r_b+1):
            li = self.cal_v(i, i)
            if li == 0:
                continue
            else:
                entropy = entropy - (li / self.v_G) * math.log(li / (gi + vi), 2)
        return entropy

    # to calculate the infomap entropy of one node
    def cal_node_infomap(self, node, tree_tmp):
        #  child_leaf for the summation of infomap entropy for all the children nodes
        #  gamma is the sum of g for all the children nodes
        #  v_G is the total volume of the graph
        #  g is the inter- connection for the partition (or called node) and should be a list containing values of g for all the children nodes
        nd_l = node.child_node_list
        child_leaf = 0
        g = node.g_i
        gamma = 0
        gc_list = []
        # the node only contains bins with no child nodes
        if node.is_leaf == 1:
            v = node.v_i
            # calculate child_leaf infomap
            child_leaf = self.cal_leaf_entro(l_b=node.left_bound, r_b=node.right_bound, gi=node.g_i, vi=node.v_i)
            if g == 0:
                infomap_node = child_leaf
            else:
                infomap_node = child_leaf - (g / self.v_G) * math.log(g / (g + v), 2)
            return infomap_node
        # calculate the value of gamma
        for i in range(len(nd_l)):
            node_c_i = nd_l[i]
            node_c = tree_tmp.node_list[node_c_i]
            gc_i = node_c.g_i
            gamma = gamma + gc_i
            gc_list.append(gc_i)
        for i in range(len(nd_l)):
            gc_i = gc_list[i]
            if gc_i == 0:
                continue
            else:
                child_leaf = child_leaf - (gc_i / self.v_G) * math.log(gc_i / (gamma + g), 2)
        # if the node is the root node
        if node.index == 0:
            infomap_node = child_leaf
        else:
            if g == 0:
                infomap_node = child_leaf
            else:
                infomap_node = child_leaf - (g / self.v_G) * math.log(g / (gamma + g), 2)
        return infomap_node

    def cal_tree_infomap(self, tree_tmp=None):
        leaf_entropy = self.cal_leaf_sum(0, self._N_-1)
        gamma = 0
        nd_l = tree_tmp.node_list[0].child_node_list
        H_ac = 0
        for i in range(len(nd_l)):
            nd_i = nd_l[i]
            nd = tree_tmp.node_list[nd_i]
            gi = nd.g_i
            vi = nd.v_i
            gamma = gamma + gi
            if gi == 0:
                g_log_g = 0
            else:
                g_log_g = gi / self.v_G * math.log(gi, 2)
            if vi + gi == 0:
                v_p_g_log = 0
            else:
                v_p_g_log = (vi + gi) / self.v_G * math.log((vi+gi), 2)
            H_ac = H_ac - 2 * g_log_g + v_p_g_log
        if gamma == 0:
            gamma_log_gamma = 0
        else:
            gamma_log_gamma = gamma / self.v_G * math.log(gamma, 2)
        total = H_ac - leaf_entropy + gamma_log_gamma
        return total

    # def cal_t_ac(self, i, i_r, t):
    #     if i >= i_r:
    #         print("ERROR! No other nodes needed!")
    #     t_left = (t * self.step_size - (self.cal_g(i + 1, i_r))) / self.step_size
    #     if t_left < 0:
    #         # This situation is impossible.
    #         # print("No t left!")
    #         return None
    #     # # looking for the most close to the true gamma
    #     # # t_left may not be an integer
    #     # directly
    #     t_ac = int(t_left + 0.5)
    #     # t_ac = t_left * 10 / 10 if t_left * 10 % 10 < 5 else t_left * 10 / 10 + 1
    #     if t_ac < 0:
    #         print("===============being stuck===================")
    #         pdb.set_trace()
    #     return t_ac

    def cal_t(self, i, i_r, t_ac):
        if i >= i_r:
            print("ERROR! No other nodes needed!")
        t_total = (t_ac * self.step_size + (self.cal_g(i + 1, i_r))) / self.step_size
        if t_total > self.l_max:
            return None
        else:
            # round half
            t = int(t_total + 0.5)
        if t > self.l_max:
            pdb.set_trace()
        return t

    def dp_process_one_layer(self):
        if self.k_max is None:
            # max_k = int(self.n / int(params['min_size_of_TAD']))
            self.k_max = int(self._N_/2)
        # when k = 1, only one tree node
        for j in range(self._N_):
            H_t = self.cal_delta(0, j)
            gamma_true = self.cal_g(0, j)
            t_c = gamma_true / self.step_size
            if abs(t_c - int(t_c)) <= abs(int(t_c) + 1 - t_c):
                t = int(t_c)
            else:
                t = int(t_c) + 1
            self.dp_table[j, t, 0] = H_t
        print(self.dp_table[:, :, 0])
        # # one node
        # tree_tmp = MT.MultiTree(n=self._N_, v=self.v_G)
        # org_bound = (list(range(self._N_)), list(range(self._N_)))
        # self.bound_list.append(org_bound)
        # tree_index = 0
        # g_list = self.cal_g_list(tree_index)
        # v_list = self.cal_v_list(tree_index)
        # tree_tmp.add_one_layer(bound=self.bound_list[tree_index], g_list=g_list, v_list=v_list)
        # # build the first tree
        # entro_tmp = self.cal_tree_infomap(tree_tmp)
        # del tree_tmp
        # self.dp_table[self._N_-1,:0] = entro_tmp
        # self.bound_list=[]
        # copy the array to the final table
        # del dp_table
        # dp_table = np.zeros(shape=(l_max + 1, self._N_))
        for k in range(1, self.k_max, 1):
            # dp_table = np.zeros(shape=(math.ceil(self.v_G/self.step_size), self._N_))
            for j in range(self._N_):
                i_r = j
                dp_where = np.where(self.dp_table[:i_r, :, k - 1] != float("inf"))
                # if k == 2 and j == self._N_-1:
                #     pdb.set_trace()
                for dp_i in range(len(dp_where[0])):
                    i = dp_where[0][dp_i]
                    t_ac = dp_where[1][dp_i]
                    t = self.cal_t(i=i, i_r=i_r, t_ac=t_ac)
                    if t is None:
                        continue
                    H_i_km1 = self.dp_table[i, t_ac, k - 1]
                    tmp = H_i_km1 + self.cal_delta((i + 1), i_r)
                    # for debug
                    # if k == 2 and i_r == 34:
                    #     print("i: ",i," H_i: ",H_i_km1," delta: ", self.cal_delta((i + 1), i_r), " H_now: ", tmp)
                    if tmp < self.dp_table[i_r, t, k]:
                        self.dp_table[i_r, t, k] = tmp
                        self.table_i[i_r, t, k] = i
                        self.table_t[i_r, t, k] = t_ac
                del dp_where
                # for t in range(0, self.l_max + 1):
                #     # find out the smallest H currently with different i
                #     # self.cal_H(j, t, k)                        #aborted for saving time
                #     i_r = j
                #     min_H = float("inf")
                #     min_i = int(-1)
                #     min_t = int(-1)
                #     for i in range(i_r):
                #         t_ac = self.cal_t_ac(i, i_r, t)
                #         if t_ac is None:
                #             continue
                #         H_i_km1 = self.dp_table[i, t_ac, k - 1]
                #         if H_i_km1 == float("inf"):
                #             continue
                #         tmp = H_i_km1 + self.cal_delta((i + 1), i_r)
                #         if tmp < min_H:
                #             min_H = tmp
                #             # may be used in the future
                #             min_i = i
                #             min_t = t_ac
                #     self.dp_table[i_r, t, k] = min_H
                #     self.table_i[i_r, t, k] = min_i
                #     self.table_t[i_r, t, k] = min_t
        print(self.dp_table[:, :, self.k_max-1])

    def copy_tree(self, tree=None):
        tree_cp = copy.deepcopy(tree)
        return tree_cp

    def legal_or_not(self, var):
        if var < 0:
            return 0
        else:
            return 1

    def find_all_nodes_with_tables(self, t_f, k_max):
        bound = ([], [])
        list_start = len(self.bound_list)
        self.bound_list.append(bound)
        t_tmp = t_f
        for k in range(k_max - 1, 0, -1):
            bound = self.bound_list[list_start]
            if k != k_max - 1:
                i_r = bound[0][-1] - 1
            else:
                i_r = self._N_ - 1
            bound[1].append(i_r)
            i_pre = self.table_i[i_r, t_tmp, k]
            t_pre = self.table_t[i_r, t_tmp, k]
            if (i_pre > self._N_ or t_pre > self.l_max) and k != 1:
                pdb.set_trace()
            # suppose there is only one i in each position
            # i_r is the right bound of the former node
            t_tmp = t_pre
            i_r = i_pre
            bound[0].append(i_r + 1)
            if k == 1:
                bound[0].append(0)
                bound[1].append(i_r)
        return 1

    def cal_g_list(self, b_i):
        g_list = []
        bound = self.bound_list[b_i]
        for i in range(len(bound[0])):
            l_b = bound[0][i]
            r_b = bound[1][i]
            g_temp = self.cal_g(start=l_b, end=r_b)
            g_list.append(g_temp)
        return g_list

    def cal_v_list(self, b_i):
        v_list = []
        bound = self.bound_list[b_i]
        for i in range(len(bound[0])):
            l_b = bound[0][i]
            r_b = bound[1][i]
            v_temp = self.cal_v(start=l_b, end=r_b)
            v_list.append(v_temp)
        return v_list

    def delete_tree_attr(self, tree):
        delattr(tree, 'node_list')
        delattr(tree, 'height')
        delattr(tree, 'root')
        return 1

    def reverse_bound(self):
        # reverse all the boundaries
        for b_list_i in range(len(self.bound_list)):
            bound = self.bound_list[b_list_i]
            bound[0].reverse()
            bound[1].reverse()

    # backtrace for all the k and t
    def back_trace_all_k(self, result_list):
        for i in range(len(result_list[0])):
            k_i = result_list[0][i]
            t_i = result_list[1][i]
            self.find_all_nodes_with_tables(t_f=t_i, k_max=k_i+1)
        self.reverse_bound()
        # add the situation where k=0
        self.bound_list.append(([0] ,[self._N_-1]))
        tree_num = len(self.bound_list)
        entro_list = []
        # multiple trees with the same minimum infomap should be considered
        for tree_index in range(tree_num):
            tree_tmp = MT.MultiTree(n=self._N_, v=self.v_G)
            g_list = self.cal_g_list(tree_index)
            v_list = self.cal_v_list(tree_index)
            tree_tmp.add_one_layer(bound=self.bound_list[tree_index], g_list=g_list, v_list=v_list)
            # build the first tree
            entro_tmp = self.cal_tree_infomap(tree_tmp)
            #  ***************************for test***************************************
            # k_i = result_list[0][tree_index]
            # t_i = result_list[1][tree_index]
            # gamma = t_i * self.step_size
            # H_tmp = self.dp_table[self._N_ - 1, t_i, k_i]
            # infomap_est = self.cal_infomap_est(H_tmp=H_tmp, gamma=gamma)
            # entro_tmp = infomap_est
            #  ***************************for test***************************************
            entro_list.append(entro_tmp)
            # delete
            # self.delete_tree_attr(tree_tmp)
            # print("k: ",tree_index," entropy: ",entro_tmp, "bound: ", self.bound_list[tree_index])
            # print(entro_tmp)
            del tree_tmp
        # choose the minimum infomap
        self.min_entro = min(entro_list)
        min_idx_list = []                      # indice of the trees with minimum infomap
        opt_bound_list = []
        for i in range(len(entro_list)):
            if self.min_entro == entro_list[i]:
                min_idx = i
                min_idx_list.append(min_idx)
                opt_bound_list.append(self.bound_list[min_idx])
        print("opt list: ", opt_bound_list)
        print("min value: ", self.min_entro)
        tree_index = min_idx_list[0]
        tree_tmp = MT.MultiTree(self._N_, self.v_G)
        g_list = self.cal_g_list(tree_index)
        v_list = self.cal_v_list(tree_index)
        tree_tmp.add_one_layer(bound=self.bound_list[tree_index], g_list=g_list, v_list=v_list)
        # check the entropy of each node
        # tree_nodes = tree_tmp.node_list
        # for nd_i in range(len(tree_nodes)):
        #     nd_entro = self.cal_node_infomap(tree_nodes[nd_i], tree_tmp)
        #     print("node:", nd_i, "infomap:", nd_entro)
        return opt_bound_list

    def cal_infomap_est(self,  H_tmp, gamma):
        if gamma == 0:
            infomap_est = H_tmp - self.cal_leaf_sum(0, self._N_-1)
        else:
            infomap_est = H_tmp + gamma / self.v_G * math.log(gamma, 2) - self.cal_leaf_sum(0, self._N_-1)
        return infomap_est

    # find out the best k
    def find_opt_k(self):
        opt_k = self.k_max
        result_list = ([], [])                # k_index, t
        local_k = 2
        local_entro_queue = [float("inf"),float("inf"),float("inf")]
        for k in range(1, self.k_max):
            # find the optimal infomap with current k
            H_index = np.where(self.dp_table[self._N_ - 1, :, k] != float("inf"))
            infomap_min = float("inf")
            min_t = []
            for i in range(len(H_index[0])):
                t = H_index[0][i]
                gamma = t * self.step_size
                H_tmp = self.dp_table[self._N_ - 1, t, k]
                infomap_est = self.cal_infomap_est(H_tmp=H_tmp, gamma=gamma)
                if infomap_min > infomap_est:
                    infomap_min = infomap_est
                    min_t.clear()
                    min_t.append(t)
                elif infomap_min == infomap_est:
                    min_t.append(t)
            # print(" entropy:", infomap_min, " bound: ", result_list)
            for i in range(len(min_t)):
                result_list[0].append(k)
                result_list[1].append(min_t[i])
            # early stop
            # print(" entropy:",infomap_est," bound: ",result_list)
            # local_entro_queue.pop(0)
            # local_entro_queue.append(infomap_min)
            # if infomap_min == float("inf"):
            #     continue
            # else:
            #     # print(infomap_est)
            #     if local_entro_queue[0]>local_entro_queue[1] and local_entro_queue[2]>local_entro_queue[1] and local_entro_queue[0]!=float("inf"):
            #         break
            #     else:
            #         continue
        # calculate the actual value of infomap in the result_list
        opt_bound_list = self.back_trace_all_k(result_list)
        # if len(opt_bound_list) == 1:
        #     opt_bound = opt_bound_list[0]
        #     opt_k = len(opt_bound[0])
        print(opt_bound_list)
        return opt_bound_list

    def save_bound(self, opt_bound_list, data_file_name):
        file_name = data_file_name + "_detection" + ".txt"
        f = open(file_name, "w")
        f.close()
        with open(file_name, 'ab') as f:
            for i in range(len(opt_bound_list)):
                bound = opt_bound_list[i]
                # save the both lists in the tuple
                np.savetxt(f, bound, fmt='%i')
                # f.write("\n")
            # min_entro_value = np.zeros(shape=(1,))
            # min_entro_value[0] = self.min_entro
            # np.savetxt(f, min_entro_value)
        return 1

    def output_bounds(self, opt_bound_list, data_file_name, start_pos=1):
        file_name = "./result/" + data_file_name + ".tsv"
        # format: chr_name start_bin(index+1) (start_bin-1)*resolution start_bin*resolution chr_name end_bin (
        # index+1) (end_bin-1)*resolution end_bin*resolution form the content
        output = []
        chr_name = data_file_name[0:5]
        resolution_s = data_file_name[8:11]
        if resolution_s[-1] == 'k':
            resolution_s = resolution_s.strip("k")
        resolution = int(resolution_s)*1000
        for i in range(len(opt_bound_list)):
            bound = opt_bound_list[i]
            # start_pos = start_pos + bound[0][0]
            for b_i in range(len(bound[0])):
                line = [chr_name, (bound[0][b_i] + start_pos), (bound[0][b_i] + start_pos - 1) * resolution, (bound[0][b_i] + start_pos) * resolution,
                        chr_name, (bound[1][b_i] + start_pos), (bound[1][b_i] + start_pos - 1) * resolution, (bound[1][b_i] + start_pos) * resolution]
                output.append(line)
        # write file
        output = DataFrame(output)
        with open("./result/temp.tsv", 'w') as write_tsv:
            write_tsv.write(output.to_csv(sep='\t', index=False))
        with open("./result/temp.tsv", 'r') as f:
            with open(file_name, 'w') as f1:
                next(f)  # skip header line
                for line in f:
                    f1.write(line)
        print("write boundaries successfully: " + file_name)
        return 1

    def construct_one_layer(self, filename="simulate_data", start_pos=1):
        self.dp_process_one_layer()
        # self.back_trace(self.k_max)
        opt_bound_list = self.find_opt_k()
        opt_bound_list = self.after_construction_redirecting(opt_bound_list)
        self.opt_bound_list = opt_bound_list
        self.save_bound(opt_bound_list, filename)
        # comp.compare()
        if(filename!="simulate_data"):
            self.output_bounds(opt_bound_list, filename, start_pos)
