from sklearn.datasets import load_iris
import numpy as np
import operator


iris = load_iris()
X = iris.data[:,:2]
y = iris.target

Xy = np.column_stack((X,y))

def gini_score(groups,classes):
    n_samples = sum([len(group) for group in groups])
    gini = 0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        #print(size)
        for class_val in classes:
            #print(group.shape)
            p = (group[:,-1] == class_val).sum() / size
            score += p * p
        gini += (1.0 - score) * (size / n_samples)
        #print(gini)
    return gini

def split(feat, val, Xy):
    Xi_left = np.array([]).reshape(0,3)
    Xi_right = np.array([]).reshape(0,3)
    for i in Xy:
        #print(i.shape)
        if i[feat] <= val:
            Xi_left = np.vstack((Xi_left,i))
        if i[feat] > val:
            Xi_right = np.vstack((Xi_right,i))
    return Xi_left, Xi_right

def best_split(Xy):
    classes = np.unique(Xy[:,-1])
    best_feat = 999
    best_val = 999
    best_score = 999
    best_groups = None
    for feat in range(Xy.shape[1]-1):
        for i in Xy:
            groups = split(feat, i[feat], Xy)
            #print(groups)
            gini = gini_score(groups, classes)
            #print('feat {}, valued < {}, scored {}'.format(feat,i[feat], gini))
            if gini < best_score:
                best_feat = feat
                best_val = i[feat]
                best_score = gini
                best_groups = groups
    output = {}
    output['feat'] = best_feat
    output['val'] = best_val
    output['groups'] = best_groups
    return output

def terminal_node(group):
    classes, counts = np.unique(group[:,-1],return_counts=True)
    return classes[np.argmax(counts)]

def split_branch(node, max_depth, min_num_sample, depth):
    left_node, right_node = node['groups']
    del(node['groups'])
    if not isinstance(left_node,np.ndarray) or not isinstance(right_node,np.ndarray):
        node['left'] = node['right'] = terminal_node(left_node + right_node)
        return
    if depth >= max_depth:
        node['left'] = terminal_node(left_node)
        node['right'] = terminal_node(right_node)
        return
    if len(left_node) <= min_num_sample:
        node['left'] = terminal_node(left_node)
    else:
        node['left'] = best_split(left_node)
        split_branch(node['left'], max_depth, min_num_sample, depth+1)
    if len(right_node) <= min_num_sample:
        node['right'] = terminal_node(right_node)
    else:
        node['right'] = best_split(right_node)
        split_branch(node['right'], max_depth, min_num_sample, depth+1)

def build_tree(Xy, max_depth, min_num_sample):
    root = best_split(Xy)
    split_branch(root, max_depth, min_num_sample, 1)
    return root

def display_tree(node, depth=0):
    if isinstance(node,dict):
        print('{}[feat{} < {:.2f}]'.format(depth*'\t',(node['feat']+1), node['val']))
        display_tree(node['left'], depth+1)
        display_tree(node['right'], depth+1)
    else:
        print('{}[{}]'.format(depth*'\t', node))

Xy = np.column_stack((X,y))
tree = build_tree(Xy, 2, 30)
display_tree(tree)