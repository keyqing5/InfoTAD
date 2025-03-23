"""
2022/2/2
rewrite
"""
import math
import pdb

import src.TreeNode as node


class MultiTree:

    def __init__(self, n=0, v=0):
        self.node_list = []
        self.height = 1
        self.root = node.TreeNode(l_b=0, r_b=n-1, g=0, v=v, h=0, is_root_=1)
        self.node_list.append(self.root)

    def add_one_layer(self, bound=None, parent_node=None, g_list=None, v_list=None):
        if bound is None:
            print("ERROR! The bound list is empty!")
            return 0
        if parent_node is None:
            # the first layer
            parent_node_i = self.root.index
        for i in range(len(bound[0])):
            # for each node
            # need the left and right boundary
            s_b = bound[0][i]
            e_b = bound[1][i]
            self.insert_tree_node(start_bin=s_b, end_bin=e_b, g=g_list[i], v=v_list[i], parent_node_index=parent_node_i)

    def insert_tree_node(self, start_bin, end_bin, g=0, v=0, parent_node_index=None):
        if start_bin < 0 or end_bin < 0:
            print("ERROR: the index is negative")
            return 0
        if start_bin > end_bin:
            print("ERROR: start bin > end_bin")
            return 0
        node_num = self.get_node_num()  # keep the total number
        if parent_node_index is None:
            print("ERROR: no parent")
        # the height of the child node
        h = self.node_list[parent_node_index].height + 1
        child_node = node.TreeNode(l_b=start_bin, r_b=end_bin, g=g, v=v, node_index=node_num, h=h, parent_i=parent_node_index)   # l_b=0, r_b=0, node_index=0, h=None, parent_i=None, is_root_=
        # modify the information of original parent node
        # self.node_list[parent_node_index].add_child(child_node.index)
        self.get_parent(parent_node_index).add_child(child_node.index)
        self.node_list.append(child_node)
        return 1

    def get_node_num(self):
        return len(self.node_list)

    def get_parent(self, parent_node_index=None):
        if parent_node_index is None:
            print("ERROR: the parent node index is empty")
            return 0
        return self.node_list[parent_node_index]
