"""
This script is to simulate a matrix with pixel is one or zero.
"""

import numpy as np
from scipy.special import comb, perm    #combination & permutation
import random
import os, getopt, sys


def usage():
    print('--help help \n'
          '-h simulated height, default: 1 \n'
          '-n # of leaf nodes, default: 5 \n'
          '-i min size of TAD, default: 10\n'
          '-j max size of TAD, default: 15\n'
          '-b noise ratio, default: 0.1\n'
          '-e edge ratio, default: 0.5\n'
          '-o output path, default: /SE_HierarchyTAD/')

class Simulator():

    def __init__(self, params, dirnames, write=False, tad=None):
        self.write = write
        self.TAD_n = np.random.randint(low=int(params['min_n']), high=int(params['max_n']))
        if tad == None:
            self.tad = np.random.randint(low=int(params['min_size']), high=int(params['max_size']), size=self.TAD_n)
        else:
            self.tad = tad
        self.height = int(params['height'])    #the height of tree
        self.edge_ratio = float(params['edge_ratio'])
        self.background_ratio = self.edge_ratio * float(params['noise_ratio']) / (1 - float(params['noise_ratio']))
        print("edge_ratio={}, background_ratio={}".format(self.edge_ratio, self.background_ratio))
        # self.dict_boundary = {}
        self.list_boundary = []
        init_pos = 1
        boundary = [init_pos]  #
        for i in range(0, len(self.tad), 1):
            init_pos += self.tad[i]
            boundary.append(init_pos)

        self.Number = boundary[-1] - 1  #the size of matrix
        self.matrix = None  # matrix
        self.simulate_structure = os.path.join(dirnames['input_data'], params['simulate_structure'])
        self.simulate_matrix = os.path.join(dirnames['input_data'], params['simulate_matrix'])
        for i in range(0, len(boundary)-1, 1):
            self.list_boundary.append([boundary[i], boundary[i+1]-1])
            # self.dict_boundary[str(boundary[i])+'-'+str(boundary[i+1]-1)] = 1

        self.boundary = boundary  # the boundary position list of TADs (containing the starting and ending)
        # print('Defined number of leaves ', self.TAD_n)
        print(self.boundary)
        if self.write:
            with open(self.simulate_structure, "w") as output:
                for i in range(0, len(self.boundary) - 1, 1):
                    for j in range(self.boundary[i], self.boundary[i + 1], 1):
                        output.writelines(str(j) + " ")
                    output.write("\n")


    def generate(self, linear=False):
        if linear == False:
            if self.height == 1:
                self.matrix = self.add_background()
                self.matrix += self.generate_binary_new()
                self.matrix = np.where(self.matrix > 1, 1, self.matrix)
                self.matrix = np.triu(self.matrix)
                self.matrix += self.matrix.T - np.diag(self.matrix.diagonal())

            else:
                boundary = []

                self.matrix = self.add_background()
                self.matrix += self.generate_binary_new()
                for h in range(self.height, 1, -1):
                    self.edge_ratio -= 0.2  #
                    if(self.edge_ratio <= 0):
                        self.edge_ratio = 0
                    boundary.append(self.boundary)
                    self.boundary = self.boundary[::3]  #
                    print(self.boundary)
                    for i in range(0, len(self.boundary) - 1, 1):
                        self.list_boundary.append([self.boundary[i], self.boundary[i+1]-1])
                        # self.dict_boundary[str(self.boundary[i]) + '-' + str(self.boundary[i + 1] - 1)] = 1
                    with open(self.simulate_structure, "a") as output:
                        for i in range(0, len(self.boundary) - 1, 1):
                            for j in range(self.boundary[i], self.boundary[i + 1], 1):
                                output.writelines(str(j) + " ")
                            output.write("\n")
                    self.matrix += self.generate_binary_new()
                # self.matrix = np.where(self.matrix > 1, 1, self.matrix)
                self.matrix = np.triu(self.matrix)
                self.matrix += np.triu(self.matrix, 1).T
        else:   #linear
            boundary = []
            self.matrix = self.generate_backgroung_linear()
            self.matrix += self.generate_linear()
            for h in range(self.height, 1, -1):
                boundary.append(self.boundary)
                self.boundary = self.boundary[::3]  #
                print(self.boundary)
                for i in range(0, len(self.boundary) - 1, 1):
                    self.list_boundary.append([self.boundary[i], self.boundary[i+1]+1])
                self.matrix += self.generate_linear()
            maximum = np.max(self.matrix)
            minimum = np.min(self.matrix)
            self.matrix = (self.matrix - minimum) / (maximum - minimum)
            self.matrix = np.triu(self.matrix)
            self.matrix += np.triu(self.matrix, 1).T

        #save the matrix
        if self.write:
            np.savetxt(self.simulate_matrix, self.matrix)
        self.matrix[self.matrix>0] = 1  ###
        return self.matrix, self.list_boundary, self.TAD_n

    def generate_binary_new(self):
        """
        Construct a matrix from combination of clustering (with different levels).
        :return:
        """
        matrix = np.zeros((self.Number, self.Number))
        for i in range(0, len(self.boundary) - 1, 1):  # if gap existing, assuming TAD and gap separate from each other
            # Add interaction within TAD
            TAD_size = self.boundary[i + 1] - self.boundary[i]
            edge_num = int((comb(TAD_size, 2) - TAD_size + 1) * self.edge_ratio)
            in_list = []
            for j in range(self.boundary[i], self.boundary[i + 1] - 2):
                for k in range(j + 2, self.boundary[i + 1]):
                    in_list.append([j - 1, k - 1])
            in_coordinate = random.sample(in_list, edge_num)
            for j in range(0, edge_num):
                matrix[in_coordinate[j][0], in_coordinate[j][1]] = 1  # np.random.randint(1, 10, 1)
        return matrix

    def add_background(self):
        matrix = np.zeros((self.Number, self.Number))
        # pixel in diagonal and second diagonal set to 1
        for i in range(0, self.Number - 1):
            matrix[i, i] = 1  # 10
            matrix[i, i + 1] = 1  # np.random.randint(1, 10, 1)
        matrix[self.Number - 1, self.Number - 1] = 1  # 10
        # add background layer
        in_list_total = []
        for j in range(1, self.Number - 2):
            for k in range(j + 2, self.Number):
                in_list_total.append([j - 1, k - 1])
        edge_num_total = int((comb(self.Number, 2) - self.Number + 1) * self.background_ratio)
        in_coordinate = random.sample(in_list_total, edge_num_total)
        for j in range(0, edge_num_total):
            matrix[in_coordinate[j][0], in_coordinate[j][1]] = 1
        return matrix


    def generate_linear(self):
        """
        Generate HiC matrix with two linear function representing TAD and background.
        :return:
        """
        matrix = np.zeros((self.Number, self.Number))
        tad_slope = -1
        tad_intercept = 100
        for i in range(0, len(self.boundary)-1, 1):
            for m in range(self.boundary[i], self.boundary[i + 1]):
                for n in range(m+1, self.boundary[i+1]):
                    matrix[m-1][n-1] = tad_slope*(n-m) + tad_intercept

        # matrix = np.triu(matrix)
        # matrix += matrix.T - np.diag(matrix.diagonal())
        # maximum = np.max(matrix)
        # minimum = np.min(matrix)
        # matrix = (matrix-minimum)/(maximum-minimum)

        return matrix

    def generate_backgroung_linear(self):
        matrix = np.zeros((self.Number, self.Number))
        b_slope = -0.02
        b_intercept = 10
        for m in range(0, self.Number, 1):
            for n in range(m, self.Number, 1):
                matrix[m][n] = b_slope*(n-m) + b_intercept
        # maximum = np.max(matrix)
        # minimum = np.min(matrix)
        # matrix = (matrix-minimum)/(maximum-minimum)
        return matrix

def main(argv=None):
    try:
        opts, args = getopt.getopt(argv, "h:n:i:j:b:e:o:", ["help"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    simu_dist = {
        'min_n': 5,
        'max_n': 6,
        'min_size': 10,
        'max_size': 15,
        'height': 1,
        'noise_ratio': 0.1,
        'edge_ratio': 0.5,
        'simulate_structure': 'simulate_structure.txt',
        'simulate_matrix': 'simulate_matrix.txt'
    }
    dirnames = {'input_data': "./"}
    for op, value in opts:
        if op == "-h":
            simu_dist['height'] = int(value)
        elif op == "-n":
            simu_dist['min_n'] = int(value)
            simu_dist['max_n'] = int(value)+1
        elif op == "-i":
            simu_dist['min_size'] = int(value)
        elif op == "-j":
            simu_dist['max_size'] = int(value)
        elif op == "-b":
            simu_dist['noise_ratio'] = float(value)
        elif op == "-e":
            simu_dist['edge_ratio'] = float(value)
        elif op == "-o":
            dirnames['input_data'] = value
        elif op == "--help":
            usage()
            sys.exit()

    matrix, realBoun, _ = Simulator(simu_dist, dirnames, write=True).generate()


def run_for_script(edge_ratio=0, min_n=5, max_n=6, min_size=10, max_size=15, noise_ratio=0.1, height=1, write=True):
    simu_dist = {
        'min_n': min_n,
        'max_n': max_n,
        'min_size': min_size,
        'max_size': max_size,
        'height': height,
        'noise_ratio': noise_ratio,
        'edge_ratio': edge_ratio,
        'simulate_structure': 'simulate_structure.txt',
        'simulate_matrix': 'simulate_matrix.txt'
    }
    dirnames = {'input_data': "./"}
    matrix, realBoun, _ = Simulator(simu_dist, dirnames, write=write).generate()
    return matrix, realBoun
  
# if __name__ == "__main__":
#     main(sys.argv[1:])

# run_for_script(edge_ratio=0.6, min_n=4, max_n=5, min_size=10, max_size=11, noise_ratio=0.2, write=True)