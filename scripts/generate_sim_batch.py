"""
2024/06/19
run by the shell file
"""
import sys, getopt
sys.path.append("..")
from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np
import time
import os

from utils import simulate_data as sim

import pdb


def write_log(words):
     with open("comparison_log.txt", 'a') as f:
         content = words + '\n'
         f.write(content)

def bound2domain(bound, key_word="sim",filename="sim.doimains"):
    with open(filename, "w") as file:
        for i in range(len(bound)):
            line = f"{key_word} {bound[i][0]} 0 0 {key_word} {bound[i][1]}\n"
            file.write(line)

def generate_data(noise_ratio=0.1,min_n=5, max_n=6, min_size=10, max_size=15, edge_ratio=0.0, times=10, file_name='', KRnorm=False, save_dir=None):
    save_mat_dir = save_dir + "/" + file_name + "/" + "mat/"
    if not save_mat_dir is None and not os.path.exists(save_mat_dir):
        os.makedirs(save_mat_dir)
    save_gt_dir = save_dir + "/" + file_name + "/" + "gt/"
    if not save_gt_dir is None and not os.path.exists(save_gt_dir):
        os.makedirs(save_gt_dir)
    save_tuple_dir = save_dir + "/" + file_name + "/" + "tuple/"
    if not save_tuple_dir is None and not os.path.exists(save_tuple_dir):
        os.makedirs(save_tuple_dir)
    for t in range(times):
        filename = save_mat_dir + file_name + "_no_"+ str(t) +".txt"
        input_matrix, bound = sim.run_for_script(edge_ratio=edge_ratio, noise_ratio=noise_ratio, min_n=min_n,
                                                    max_n=max_n, min_size=min_size,
                                                    max_size=max_size, write=False)
        # save
        # save mat
        np.savetxt(filename, input_matrix)
        # save bound as domains format
        domain_file = save_gt_dir + file_name + "_no_"+ str(t) +".domains"
        bound2domain(bound=bound, filename=domain_file)
        # save in tuple format
        tuple_file = save_tuple_dir + file_name + "_no_"+ str(t) +".txt"
        with open(tuple_file, "w") as file:
            for i in range(len(bound)):
                line = f"{bound[i][0]} {bound[i][1]}\n"
                file.write(line)


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "N:M:m:S:s:e:T:K:D:")
    except getopt.GetoptError:
        sys.exit(2)
    min_n = 3
    max_n = 5
    edge_ratio = 0.6
    min_size = 10
    max_size = 11
    times = 10
    noise_ratio = 0.15
    KRnorm = False
    for op, value in opts:
        if op == '-N':
            noise_ratio = float(value)
        elif op == "-M":
            max_n = int(value)
        elif op == "-m":
            min_n = int(value)
        elif op == "-S":
            max_size = int(value)
        elif op == "-s":
            min_size = int(value)
        elif op == "-e":
            edge_ratio = float(value)
            # edge_ratio = edge_ratio * 0.1
        elif op == "-T":
            times = int(value)
        elif op == "-D":
            save_dir = value
        elif op == "-K":
            KRnormint = int(value)
            if KRnormint == 1:
                KRnorm = True
            else:
                KRnorm = False
        else:
            print("option error!")
            sys.exit()

    if KRnorm:
        file_name = '_edge=' + str(edge_ratio) + '_noise' + str(noise_ratio) + '_m=' + str(min_n)+'_M'+str(max_n) + '_s=' + str(
            min_size)+'_S'+str(max_size) + '_T' + str(times)+"KR"
    else:
        file_name = '_edge='+str(edge_ratio)+'_noise'+ str(noise_ratio)+'_m='+str(min_n)+'_M'+str(max_n)+'_s='+str(min_size)+'_S'+str(max_size)+'_T'+str(times)
    generate_data(noise_ratio=noise_ratio,min_n=min_n, max_n=max_n, min_size=min_size, max_size=max_size, edge_ratio=edge_ratio, times=times, file_name=file_name, KRnorm=KRnorm, save_dir=save_dir)



if __name__ == "__main__":
    main(sys.argv[1:])
