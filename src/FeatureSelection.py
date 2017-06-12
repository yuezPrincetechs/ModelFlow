# -*- coding: utf-8 -*-
'''
变量筛选模块。
'''

import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
import copy
from sklearn import feature_selection
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection.base import SelectorMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

class BaseSelector(BaseEstimator,TransformerMixin):
    '''
    抽象类，用于自定义扩展变量筛选方法。
    
    基本属性: 
    feature_selected: 数值或字符串列表，表示筛选出来的特征所在列号或列名。
    return_array: True则统一返回numpy.ndarray，Fasle则返回形式同X。
    '''
    __metaclass__=ABCMeta
    
    def __init__(self,return_array=False):
        self.return_array=return_array
    
    @abstractmethod
    def fit(self,X,y):
        '''
        训练模型，获取筛选的变量列表。
        X: dataframe或二维数组。
        Y: series或一维数组。
        '''
        pass
    
    def transform(self,X):
        '''
        根据筛选得到的变量转换数据。
        X: dataframe或二维数组。
        返回数据形式同X。
        '''
        if isinstance(X,np.ndarray):
            return X[:,self.feature_selected].copy()
        else:
            if self.return_array:
                return X[self.feature_selected].copy().values
            else:
                return X[self.feature_selected].copy()
    
    def fit_transform(self,X,y):
        '''
        训练模型，获取筛选的变量列表并转换数据。
        X: dataframe或二维数组。
        Y: series或一维数组。
        返回数据形式同X。
        '''
        return self.fit(X,y).transform(X)


class SklearnSelector(BaseSelector):
    def __init__(self,selector,return_array=False):
        '''
        基于sklearn的特征筛选模型进行特征筛选，无缝连接sklearn。
        selector: sklearn.feature_selection模块下的筛选模型对象，如sklearn.feature_selection.SelectKBest(sklearn.feature_selection.f_classif,k=4)。
        return_array: True则统一返回numpy.ndarray，Fasle则返回形式同X。

        额外属性: 
        selector_: fit后的selector。
        '''
        BaseSelector.__init__(self,return_array=return_array)
        self.selector=selector
    
    def fit(self,X,y):
        self.selector_=clone(self.selector)
        self.selector_.fit(X,y)
        self.feature_selected=self.selector_.get_support(indices=True).tolist()
        if isinstance(X,pd.DataFrame):
            self.feature_selected=X.columns[self.feature_selected].tolist()
        return self

class VotingSelector(BaseSelector):
    def __init__(self,selectors,threshold,weights=None,return_array=False):
        '''
        根据多个筛选模型的结果投票进行特征选择。
        selectors: 筛选器列表，形式为[('name1',selector1),('name2',selector2),('name3',selector3)]。
                   其中筛选器可以是sklearn.feature_selection中的任意筛选器，程序会自动转换为SklearnSelector。
        threshold: 阈值，若为0到1的浮点数，则选取加权投票大于threshold的特征，若为大于1的正整数，则选取加权投票最多的前threshold个特征。
        weights: 列表，对应每个筛选模型的权重，None默认权重均为1。
        return_array: True则统一返回numpy.ndarray，Fasle则返回形式同X。
        
        额外属性: 
        named_selectors: 筛选器字典，键为名称，值为筛选器对象。
        df_voting: dataframe，index为变量名或列号，columns为筛选器名称，值为0（未被选中）或1（被选中）。
        score: series，index为变量名或列号，值为根据df_voting和weights计算得到的得分。
        '''
        BaseSelector.__init__(self,return_array=return_array)
        self.named_selectors=dict(selectors)
        for name in self.named_selectors:
            if not isinstance(self.named_selectors[name],BaseSelector):
                self.named_selectors[name]=SklearnSelector(self.named_selectors[name],return_array=self.return_array)
        self.threshold=threshold
        if weights is None:
            weights=[1./len(selectors)]*len(selectors)
        else:
            weights=[float(i)/sum(weights) for i in weights]
        self.weights=dict([(selectors[i][0],weights[i]) for i in range(len(weights))])
    
    def fit(self,X,y):
        if isinstance(X,pd.DataFrame):
            index=X.columns.tolist()
        else:
            index=list(range(X.shape[1]))
        df_voting={}
        for name in self.named_selectors:
            self.named_selectors[name].fit(X,y)
            selected={i:1 for i in self.named_selectors[name].feature_selected}
            df_voting[name]=copy.deepcopy(selected)
        df_voting=pd.DataFrame(df_voting)
        df_voting=df_voting.reindex_axis(index,axis=0)
        self.df_voting=df_voting.fillna(0)
        self.score=(self.df_voting*pd.Series(self.weights)).sum(axis=1)
        self.score=self.score.sort_values(ascending=False)
        if self.threshold<=1:
            self.feature_selected=self.score.index[self.score>self.threshold].tolist()
        else:
            self.feature_selected=self.score.head(self.threshold).index.tolist()
        return self
        
        




#==============================================================================
# 测试
#==============================================================================
def test():
    #生成测试数据
    np.random.seed(13)
    X=pd.DataFrame(np.random.randn(20,10))
    X.columns=['x%d'%i for i in range(10)]
    y=pd.Series(np.random.choice([0,1],20))
    
    #基于sklearn的特征筛选模型进行特征筛选
    clf_sklearn=feature_selection.SelectKBest(feature_selection.f_classif,k=4)
    clf=SklearnSelector(estimator=clf_sklearn)
    clf.fit(X,y)
    clf.transform(X)
    print(clf.feature_selected)
    
    clf_sklearn=SelectFromModel(LogisticRegression())
    clf=SklearnSelector(estimator=clf_sklearn)
    clf.fit(X,y)
    clf.transform(X)
    print(clf.feature_selected)
    
    #投票筛选器
    clf_selectkbest=feature_selection.SelectKBest(feature_selection.f_classif,k=4)
    clf_selectfrommodel=SelectFromModel(LogisticRegression())
    clf_baseselector=SklearnSelector(clf_selectkbest)
    clf=VotingSelector(selectors=[('clf_selectkbest',clf_selectkbest),
                                  ('clf_selectfrommodel',clf_selectfrommodel),
                                  ('clf_baseselector',clf_baseselector)],threshold=0.5)
    clf.fit(X,y)
    clf.transform(X)
    print(clf.feature_selected)
    print(clf.df_voting)
    print(clf.score)

