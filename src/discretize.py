# -*- coding: utf-8 -*-
"""
单变量离散化模块。
主要包括: 
    分位数离散化
    等分点离散化
    标准差离散化
    决策树离散化
"""
import numpy as np
import pandas as pd
import sklearn
import sklearn.tree
from sklearn.base import BaseEstimator,TransformerMixin
import copy
from abc import ABCMeta, abstractmethod
import warnings

class BaseDiscretizer(BaseEstimator,TransformerMixin):
    """
    抽象类，用于自定义扩展单变量离散化方法。
    必有的属性: 
    feature_names: 需要离散化的变量列表，非负整数或者字符串。
    fill_na: 对于缺失值指定特殊类别。
    return_numeric: True则返回数值类别，False则返回字符串类别。
    return_array: True则统一返回numpy.ndarray，Fasle则返回形式同X。
    decimal: 字符串类别中小数保留位数。
    cuts: 若fit的X为一维数组或Series，cuts则为分割点的一维数组（表示只有单一变量）；
          若fit的X为二维数组或DataFrame，cuts则为字典，键为变量名，值为分割点的一维数组。
    """

    __metaclass__ = ABCMeta

    def __init__(self,feature_names=None,fill_na=-1,return_numeric=True,return_array=False,decimal=2):
        self.feature_names=feature_names
        self.fill_na=fill_na
        self.return_numeric=return_numeric
        self.return_array=return_array
        self.decimal=decimal
    
    
    @abstractmethod
    def fit(self,X,y=None):
        '''
        对feature_names中的变量获取各自离散化分割点。
        X: 一维或二维数组，或DataFrame，或Series。
        y: 一维数组，或Series。
        '''
        pass

    def transform(self,X):
        """
        离散化数据: 数值为0到n-1，对于缺失值会增加类别-1。
        X: 一维或二维数组，或DataFrame，或Series。
        返回离散化后的数据，一维或二维数组，或DataFrame，或Series。
        """
        data=X.copy()
        if isinstance(data,np.ndarray):
            if isinstance(self.fill_na,str):
                raise Exception('numpy数组缺失值只能设置成数值！')
            if not self.return_numeric:
                warnings.warn('numpy数组只能返回数值编码，若想返回字符串编码，请输入dataframe或series！')
        if not self.return_numeric:
            newlabel=self.get_label()
        if len(data.shape)==1:
            tmp=np.searchsorted(self.cuts,data).astype(int)
            result=np.where(np.isnan(data),-1,tmp)
            if (not self.return_numeric) and (not isinstance(data,np.ndarray)):
                f=np.frompyfunc(lambda xx: newlabel.get(xx,self.fill_na),1,1)
                result=f(result)
            if isinstance(data,np.ndarray):
                result[result==-1]=self.fill_na
            else:
                result=pd.Series(result)
                result.index=data.index
                result.index.name=data.index.name
                result.name=data.name
                result[result==-1]=self.fill_na
            data=result.copy()
        else:
            for feature in self.cuts:
                if not isinstance(data,pd.DataFrame):
                    tmp=np.searchsorted(self.cuts[feature],data[:,feature]).astype(int)
                    data[:,feature]=np.where(np.isnan(data[:,feature]),self.fill_na,tmp)
                else:
                    tmp=np.searchsorted(self.cuts[feature],data[feature]).astype(int)
                    data[feature]=np.where(np.isnan(data[feature]),-1,tmp)
                    if not self.return_numeric:
                        f=np.frompyfunc(lambda xx: newlabel[feature].get(xx,self.fill_na),1,1)
                        data[feature]=f(data[feature])
                    else:
                        data.loc[data[feature]==-1,feature]=self.fill_na
        if self.return_array and isinstance(data,(pd.Series,pd.DataFrame)):
            return data.values
        else:
            return data
    
    def get_label(self):
        '''
        根据分割点获取类别的字符串编码。
        返回形式同视self.cuts的形式: 
        若self.cuts为一维数组（表示单变量），则返回结果为单一字典，键为数值编码（0到n-1），值为对应的分割方式字符串编码。
        若self.cuts为字典（表示多个变量），则返回结果为映射字典的字典，外层字典的键为变量名，值为映射字典，该映射字典同单变量的编码映射字典。
        例子:
            self.cuts=np.array([1,2,3])
            返回值为{0:'(-inf,1]',1:'(1,2]',2:'(2,3]',3:'(3,inf)'}
        '''
        newlabel={}
        if not isinstance(self.cuts,dict):
            newlabel=self.get_label_single(self.cuts,self.decimal)
        else:
            for feature in self.cuts:
                tmp=self.get_label_single(self.cuts[feature],self.decimal)
                newlabel[feature]=tmp.copy()
        return newlabel
    
    @staticmethod
    def get_label_single(cut,decimal):
        '''
        获取单个变量的类别字符串编码字典。
        cut: 单个变量的分割点一维数组。
        decimal: 字符串编码中小数保留位数。
        返回字典，键为数值编码（0到n-1），值为对应的分割方式字符串编码。
        '''
        newlabel={}
        newlabel[0]='(-inf,%.*f]'%(decimal,cut[0])
        if cut.shape[0]==1:
            pass
        else:
            for i in range(cut.shape[0]-1):
                newlabel[i+1]='(%.*f,%.*f]'%(decimal,cut[i],decimal,cut[i+1])
        newlabel[cut.shape[0]]='(%.*f,inf)'%(decimal,cut[-1])
        return newlabel
        


class QuantileDiscretizer(BaseDiscretizer):
    def __init__(self,feature_names=None,quantiles=[25,50,75],fill_na=-1,return_numeric=True,return_array=False,decimal=2):
        '''
        根据样本分位数离散化。
        feature_names: 需要离散化的变量列表，非负整数或者字符串。
        quantiles: 离散化的分位点列表。
        '''
        BaseDiscretizer.__init__(self,feature_names=feature_names,fill_na=fill_na,return_numeric=return_numeric,return_array=return_array,decimal=decimal)
        self.quantiles=quantiles
    
    def fit(self,X,y=None):
        '''
        对feature_names中的变量获取各自离散化分割点。
        X: 一维或二维数组，或DataFrame，或Series。
        y: 一维数组，或Series。
        '''
        if len(X.shape)==1:
            x=X[np.isnan(X)==False]
            self.cuts=np.percentile(x,self.quantiles)
        else:
            self.cuts=dict()
            if self.feature_names is None:
                try:
                    feature_names=list(X.columns)
                except:
                    feature_names=list(range(X.shape[1]))
            else:
                feature_names=self.feature_names
            for feature in feature_names:
                try:
                    x=X[:,feature].copy()
                except:
                    x=X[feature].copy()
                x=x[np.isnan(x)==False]
                self.cuts[feature]=np.percentile(x,self.quantiles)
        return self

class SimpleBinsDiscretizer(BaseDiscretizer):
    def __init__(self,feature_names=None,bins=10,fill_na=-1,return_numeric=True,return_array=False,decimal=2):
        '''
        根据等分点离散化。
        feature_names: 需要离散化的变量列表，非负整数或者字符串。
        bins: 等分区间数。
        '''
        BaseDiscretizer.__init__(self,feature_names=feature_names,fill_na=fill_na,return_numeric=return_numeric,return_array=return_array,decimal=decimal)
        if bins<=1:
            raise Exception('bins必须大于1！')
        self.bins=bins
    
    def fit(self,X,y=None):
        '''
        对feature_names中的变量获取各自离散化分割点。
        X: 一维或二维数组，或DataFrame，或Series。
        y: 一维数组，或Series。
        '''
        if len(X.shape)==1:
            vmax=X.max()
            vmin=X.min()
            if vmin==vmax:
                self.cuts=np.array([vmin])
            else:
                self.cuts=np.array([vmin+i*(vmax-vmin)/float(self.bins) for i in range(1,self.bins)])
        else:
            self.cuts=dict()
            if self.feature_names is None:
                try:
                    feature_names=list(X.columns)
                except:
                    feature_names=list(range(X.shape[1]))
            else:
                feature_names=self.feature_names
            for feature in feature_names:
                try:
                    x=X[:,feature]
                except:
                    x=X[feature]
                vmax=x.max()
                vmin=x.min()
                if vmin==vmax:
                    self.cuts[feature]=np.array([vmin])
                else:
                    self.cuts[feature]=np.array([vmin+i*(vmax-vmin)/float(self.bins) for i in range(1,self.bins)])
        return self

class MeanStdDiscretizer(BaseDiscretizer):
    def __init__(self,feature_names=None,fill_na=-1,return_numeric=True,return_array=False,decimal=2):
        '''
        根据标准差离散化，分割点为: mu-2*sigma,mu-sigma,mu-0.5*sigma,mu,mu+0.5*sigma,mu+sigma,mu+2*sigma。
        feature_names: 需要离散化的变量列表，非负整数或者字符串。
        '''
        BaseDiscretizer.__init__(self,feature_names=feature_names,fill_na=fill_na,return_numeric=return_numeric,return_array=return_array,decimal=decimal)
    
    def fit(self,X,y=None):
        '''
        对feature_names中的变量获取各自离散化分割点。
        X: 一维或二维数组，或DataFrame，或Series。
        y: 一维数组，或Series。
        '''
        if len(X.shape)==1:
            x=X[np.isnan(X)==False]
            mu=x.mean()
            sigma=x.std()
            if not (0<sigma<np.inf):
                self.cuts=np.array([mu])
            else:
                self.cuts=np.array([mu-2*sigma,mu-sigma,mu-0.5*sigma,mu,mu+0.5*sigma,mu+sigma,mu+2*sigma])
        else:
            self.cuts=dict()
            if self.feature_names is None:
                try:
                    feature_names=list(X.columns)
                except:
                    feature_names=list(range(X.shape[1]))
            else:
                feature_names=self.feature_names
            for feature in feature_names:
                try:
                    x=X[:,feature]
                except:
                    x=X[feature]
                x=x[np.isnan(x)==False]
                mu=x.mean()
                sigma=x.std()
                if not (0<sigma<np.inf):
                    self.cuts[feature]=np.array([mu])
                else:
                    self.cuts[feature]=np.array([mu-2*sigma,mu-sigma,mu-0.5*sigma,mu,mu+0.5*sigma,mu+sigma,mu+2*sigma])
        return self


class EntropyDiscretizer(BaseDiscretizer):
    def __init__(self,feature_names=None,max_depth=3,fill_na=-1,return_numeric=True,return_array=False,decimal=2,**kwds):
        '''
        根据单变量决策树离散化。
        feature_names: 需要离散化的变量列表，非负整数或者字符串。
        max_depth: 决策树最大深度，用于限制离散化的区间数。
        kwds: 其他决策树参数，参考sklearn.tree.DecisionTreeClassifier。
        '''
        BaseDiscretizer.__init__(self,feature_names=feature_names,fill_na=fill_na,return_numeric=return_numeric,return_array=return_array,decimal=decimal)
        self.max_depth=max_depth
        self.kwds=kwds
        
    def fit(self,X,y=None):
        '''
        对feature_names中的变量获取各自离散化分割点。
        X: 一维或二维数组，或DataFrame，或Series。
        y: 一维数组，或Series。
        '''
        if y is None:
            raise Exception('y未提供！')
        dt=sklearn.tree.DecisionTreeClassifier(criterion='entropy',max_depth=self.max_depth,**self.kwds)
        if len(X.shape)==1:
            dt.fit(X.reshape((-1,1)),y)
            cuts=getTreeSplits(dt)
            if cuts is None:
                # 如果决策树无分割点，则返回中位数作为分割点
                cuts=np.array([np.median(X)])
        else:
            cuts=dict()
            if self.feature_names is None:
                try:
                    feature_names=list(X.columns)
                except:
                    feature_names=list(range(X.shape[1]))
            else:
                feature_names=self.feature_names
            for feature in feature_names:
                try:
                    x=X[:,feature]
                except:
                    x=X[feature]
                x=x.reshape((-1,1))
                dt.fit(x,y)
                cut=getTreeSplits(dt)
                if cut is None:
                    cut=np.array([np.median(x)])
                cuts[feature]=cut.copy()
        self.cuts=copy.deepcopy(cuts)
        return self
    
def getTreeSplits(dt):
    '''
    获取单变量决策树的分割点并排序。
    dt: 单变量决策树分类器，参考sklearn.tree.DecisionTreeClassifier。
    返回一维数组或None（表示无分割点，空树）。
    '''
    cut=dt.tree_.threshold[np.where(dt.tree_.children_left>-1)]
    if cut.shape[0]==0:
        return None
    return np.sort(cut)
        

