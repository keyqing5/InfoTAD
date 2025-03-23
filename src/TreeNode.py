"""
2021/12/29
tree node for multi-ary tree
"""
import pdb


class TreeNode:

    def __init__(self, l_b=0, r_b=0, g=0, v=0, node_index=0, h=None, parent_i=None, is_root_=0):
        self.child_node_list = []
        if is_root_ == 0:
            if parent_i is None:
                print("ERROR: No parent node!")
            else:
                self.parent_node_i = parent_i
            self.is_leaf = 1
        else:
            self.is_root = 1
            self.height = 0       # the root is in the 0-th layer
            self.is_leaf = 0
        self.left_bound = l_b
        self.right_bound = r_b
        self.index = node_index
        self.g_i = g
        self.v_i = v
        if h is None:
            print("ERROR: no height")
        else:
            self.height = h

    def add_child(self, child_index):
        self.child_node_list.append(child_index)
        if self.is_leaf == 1:
            self.change_node_status()
        return 1

    def change_node_status(self):
        self.is_leaf = 0
        return 1

    def get_parent(self):
        return self.parent_node_i

