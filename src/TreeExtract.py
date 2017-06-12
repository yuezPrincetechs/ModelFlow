# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import copy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class TreeExtract():
    '''
    提取二叉树（sklearn.tree.tree.Tree对象）的节点路径信息。
    '''
    def __init__(self,estimators_,columns,decimal=5,sep=';'):
        self.estimators_=estimators_
        self.columns=columns
        self.decimal=decimal
        self.sep=sep
        self.paths=[]
        for estimator in self.estimators_:
            path_tmp=self.get_allpath(estimator.tree_,self.columns)
            self.paths.append(copy.deepcopy(path_tmp))
    
    def apply(self,X,return_numeric=True):
        '''
        获取叶子节点编号或路径名称。
        X：dataframe，其columns必须包含self.columns。
        return_numeric：True表示返回每个样本所在叶子节点编号，False则返回叶子节点所在路径名称。
        返回dataframe，其index同X的index，列号为0到n-1（n表示self.estimators_中树的数量），值为样本对应叶子节点编号或路径名称。
        '''
        result=pd.DataFrame()
        for i,estimator in enumerate(self.estimators_):
            result[i]=estimator.apply(X[self.columns])
        result.index=X.index
        result.index.name=X.index.name
        if return_numeric:
            return result
        else:
            return self.apply_leafnodes(result)
    
    def apply_leafnodes(self,leafnodes_numeric):
        '''
        获取叶子节点（实现上不限制于叶子节点）路径名称。
        leafnodes_numeric：dataframe，其columns为0到n-1（n表示self.estimators_中树的数量），值为样本对应叶子节点编号。
        返回dataframe，其index和columns同leafnodes_numeric，值为样本对应叶子节点路径名称。
        '''
        result=leafnodes_numeric.copy()
        for i in result.columns.tolist():
            result[i]=result[i].map(lambda node: self.get_pathname(self.paths[i][node]))
        return result
    
    def cal_pathlength(self,X):
        '''
        获取叶子节点路径长度。
        X：dataframe，其columns必须包含self.columns。
        返回dataframe，其index同X的index，列号为0到n-1（n表示self.estimators_中树的数量），值为样本对应叶子节点路径长度。
        '''
        n_samples=X.shape[0]
        
        n_samples_leaf=np.zeros((n_samples,len(self.estimators_)),order='f')
        depths=np.zeros((n_samples,len(self.estimators_)),order='f')
        
        for i,tree in enumerate(self.estimators_):
            leaves_index=tree.apply(X)
            node_indicator=tree.decision_path(X)
            n_samples_leaf[:,i]=tree.tree_.n_node_samples[leaves_index]
            depths[:,i]=np.asarray(node_indicator.sum(axis=1)).reshape(-1)-1
        
        depths+=_average_path_length(n_samples_leaf)
        
        result=pd.DataFrame(depths)
        result.columns=np.arange(len(self.estimators_))
        result.index=X.index
        result.index.name=X.index.name
        return result
        
        
    
    def count_features(self,X,verbose=False):
        '''
        对每个样本，计算所在叶子节点路径上每个特征出现的次数。
        X：dataframe，其columns必须包含self.columns。
        返回列表，每个元素对应self.estimators_中的树，每个元素均为dataframe，index同X，columns为self.columns，值为出现次数。
        '''
        result=[]
        for i,estimator in enumerate(self.estimators_):
            tmp=pd.Series(estimator.apply(X[self.columns]))
            tmp.index=X.index
            tmp=tmp.map(lambda xx: ' '.join([yy[0] for yy in self.paths[i][xx]]))
            vect=CountVectorizer(vocabulary=self.columns,lowercase=False)
            tmp=vect.transform(tmp).toarray()
            tmp=pd.DataFrame(tmp)
            vocabulary_inverse={vect.vocabulary_[key]:key for key in vect.vocabulary_}
            tmp.columns=[vocabulary_inverse[k] for k in range(tmp.shape[1])]
            tmp.index=X.index
            tmp.index.name=X.index.name
            tmp=tmp.fillna(0)
            result.append(tmp.copy())
            if verbose:
                print('Done:',i)
        return result
    
    
    def get_pathname(self,path):
        '''
        获取路径名称。
        path：路径列表，形如[(col1,threshold1,op1),(col2,threshold2,op2),...]，参见get_nodepath输出。
        返回路径名称字符串，形如'x1<=1;x2>5'。
        '''
        return self.sep.join(map(lambda xx: '%s%s%.*f'%(xx[0],xx[2],self.decimal,xx[1]),path))
    
    def get_allpath(self,tree_,columns):
        '''
        获取整棵树所有节点路径。
        tree_：sklearn.tree.tree.Tree对象。
        columns：变量名列表。
        返回字典，键为节点编号，值为节点路径，参加get_nodepath输出结果。
        '''
        paths={}
        for node_index in range(tree_.node_count):
            path=self.get_nodepath(tree_,node_index,columns)
            paths[node_index]=path.copy()
        return paths
    
    def get_nodepath(self,tree_,node_index,columns):
        '''
        获取树节点路径。
        tree_：sklearn.tree.tree.Tree对象。
        node_index：节点编号。
        columns：变量名列表。
        返回节点路径，以列表表示，形如[(col1,threshold1,op1),(col2,threshold2,op2),...]，
        其中col表示节点变量名，threshold表示分割点，op表示操作符（{'<=','>'}）。顺序同树的结构自上而下。
        若为根节点，则返回空列表。
        '''
        if node_index==0:
            return []
        try:
            node_index_parent=np.where(tree_.children_left==node_index)[0][0]
            op_parent='<='
        except:
            node_index_parent=np.where(tree_.children_right==node_index)[0][0]
            op_parent='>'
        threshold_parent=tree_.threshold[node_index_parent]
        col_parent=columns[tree_.feature[node_index_parent]]
        path=self.get_nodepath(tree_,node_index_parent,columns)
        path.append((col_parent,threshold_parent,op_parent))
        return path.copy()





def _average_path_length(n_samples_leaf):
    """ The average path length in a n_samples iTree, which is equal to
    the average path length of an unsuccessful BST search since the
    latter has the same structure as an isolation tree.
    Parameters
    ----------
    n_samples_leaf : array-like of shape (n_samples, n_estimators), or int.
        The number of training samples in each test sample leaf, for
        each estimators.

    Returns
    -------
    average_path_length : array, same shape as n_samples_leaf

    """
    import numbers
    INTEGER_TYPES = (numbers.Integral, np.integer)
    if isinstance(n_samples_leaf, INTEGER_TYPES):
        if n_samples_leaf <= 1:
            return 1.
        else:
            return 2. * (np.log(n_samples_leaf) + 0.5772156649) - 2. * (
                n_samples_leaf - 1.) / n_samples_leaf

    else:

        n_samples_leaf_shape = n_samples_leaf.shape
        n_samples_leaf = n_samples_leaf.reshape((1, -1))
        average_path_length = np.zeros(n_samples_leaf.shape)

        mask = (n_samples_leaf <= 1)
        not_mask = np.logical_not(mask)

        average_path_length[mask] = 1.
        average_path_length[not_mask] = 2. * (
            np.log(n_samples_leaf[not_mask]) + 0.5772156649) - 2. * (
                n_samples_leaf[not_mask] - 1.) / n_samples_leaf[not_mask]

        return average_path_length.reshape(n_samples_leaf_shape)










