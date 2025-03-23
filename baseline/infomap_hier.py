"""
hierarchically pick the best pair
1. read file
2. try all the possible pairs, save the maximum value of decreasing and make the pair true (actually combine,
delete the original objects, add the new pair to the list with a correct position)
3. repeat 2 until there is no way to decrease the entropy
"""
import numpy as np
import copy
import pdb
import time

import sys
sys.path.append("..")
import src.detector_v2 as detec
import src.multiTree as MT



# return the best place to combine
def try_pairs(org_bound=None, org_entro=float("inf"), detect=None):
    flag = 0
    left_i = None
    min_entro = org_entro
    # combine each two nodes until the end of the list
    max_i = len(org_bound[0]) - 1
    for i in range(max_i):
        bound_temp = make_new_bound(org_bound, i)
        entro_temp = construct_a_tree(detect, bound_temp)
        del bound_temp
        if entro_temp < min_entro:
            flag = 1
            min_entro = entro_temp
            left_i = i
    # keep the minimum value, compare with the input org_entro, if smaller flag=1, else flag=0
    if flag:
        # there is a better choice than the original
        new_bound = make_new_bound(org_bound, left_i)
        return new_bound
    else:
        # no better choice
        return None


def make_new_bound(bound=None, left_i=None):
    new_bound = copy.deepcopy(bound)
    del new_bound[0][left_i+1]
    del new_bound[1][left_i]
    return new_bound


def construct_a_tree(detect=None, bound=None):
    detect.bound_list.clear()
    detect.bound_list.append(bound)
    tree_index = 0
    tree_tmp = MT.MultiTree(n=detect._N_, v=detect.v_G)
    g_list = detect.cal_g_list(tree_index)
    v_list = detect.cal_v_list(tree_index)
    tree_tmp.add_one_layer(bound=detect.bound_list[tree_index], g_list=g_list, v_list=v_list)
    # build the first tree
    entro_tmp = detect.cal_tree_infomap(tree_tmp)
    del tree_tmp
    return entro_tmp


def find_optimal_pair(input_matrix=None, KRnorm=False):
    detect = detec.TADDetector(input_matrix, step_scal=1, KR_norm=KRnorm)
    __N__ = len(detect.matrix)
    org_bound = (list(range(__N__)), list(range(__N__)))
    # compute the original value of entropy
    org_entro = construct_a_tree(detect, org_bound)
    # combine pairs until no new pairs
    flag = 1
    while flag:
        better_bound = try_pairs(org_bound, org_entro, detect)
        # print("better: ",better_bound," entro: ",construct_a_tree(detect,better_bound))
        # pdb.set_trace()
        if better_bound is None:# or len(org_bound[0])==2:
            # no better choice
            flag = 0
            break
        else:
            del org_bound
            org_entro = construct_a_tree(detect, better_bound)
            org_bound = copy.deepcopy(better_bound)
            del better_bound
    print("hier: ",org_entro)
    print("hier_bound: ",org_bound)
    return org_bound, org_entro


def read_matrix(name="../simulate_matrix_1.txt", dim=10):
    A = np.zeros((dim, dim), dtype=float)

    f = open(name)
    lines = f.readlines()
    A_row = 0
    for line in lines:
        list_cor = line.strip('\n').split(' ')
        A[A_row:] = list_cor[0:dim]
        A_row += 1

    return A

# input_matrix = np.loadtxt("D:/experiment/record_infomap/example/simulate_matrix_orgentro4.883900214590855_our4.888870509928176.txt", dtype=np.float, delimiter=' ')
# input_matrix = np.loadtxt("D:/experiment/SuperTAD/data/input.txt", dtype=np.float, delimiter=' ')
# input_matrix = read_matrix("D:/experiment/SuperTAD/data/input.txt",dim=9)
# input_matrix = np.loadtxt("../simulate_matrix.txt", dtype=np.float, delimiter=' ')
# # input_matrix = np.loadtxt("/public/qiusliang2/hic/chr22_KR_25kb_matrix_673_1072.txt", dtype=np.float, delimiter=' ')
# # # # # input_matrix = np.loadtxt("../data/300x0/chr19_KR_150kb_matrix.txt", dtype=np.float, delimiter=' ')
# # # # # input_matrix = np.loadtxt("../data/300x0/chr19_KR_250kb_matrix.txt", dtype=np.float, delimiter=' ')
#
# input_matrix = np.loadtxt("D:/experiment/hic data/chr22/chr22_KR_25kb_matrix_673_1072.txt", dtype=np.float, delimiter=' ')
# input_matrix = np.loadtxt("D:/experiment/hic data/chr19/chr19_KR_25kb_matrix_421_820.txt", dtype=np.float, delimiter=' ')
# # start_time = time.time()
# opt_bound, org_entro = find_optimal_pair(input_matrix,KRnorm=False)
# # end_time = time.time()
# # running_time = end_time - start_time
# # print("running time: ", running_time)
# print(opt_bound)
# print(org_entro)
# opt_bound, org_entro = find_optimal_pair(input_matrix,KRnorm=False)
# data_name = "chr22_KR_25kb_matrix_673_1072"
# file_name = data_name+"detection.txt"
# f = open(file_name, "w")
# f.close()
# with open(file_name, 'ab') as f:
# # save the both lists in the tuple
#     np.savetxt(f, opt_bound, fmt='%i')
# print(opt_bound)
# print(org_entro)
# comp.compare(hier=True)
