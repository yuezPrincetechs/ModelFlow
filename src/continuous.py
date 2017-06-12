# -*- coding: utf-8 -*-
'''
单变量连续化模块。
主要包括：
    条件概率连续化
    WOE连续化
    Label连续化编码（改写版）
    独热编码（改写版）
'''

import numpy as np
import pandas as pd
import copy
from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import src.utils.utils as utils

class BaseContinuous(BaseEstimator,TransformerMixin):
    """
    抽象类，用于自定义扩展单变量连续化方法。
    必有的属性：
    feature_names：需要连续化的变量列表，非负整数或者字符串。
    return_array: True则统一返回numpy.ndarray，Fasle则返回形式同X。
    method：具体使用的连续化方法名称。
    crosstabs：列联表或列联表字典；
               若fit的X为Series或一维数组，crosstabs则为列联表DataFrame，对应变量不同类别取值及不同响应变量下的样本数量；
               若fit的X为DataFrame或二维数组，crosstabs则为字典，键为变量名，值为列联表DataFrame。
    single：是否为单一变量。
    maps：映射字典或映射字典的字典；
          若fit的X为Series或一维数组，maps则为映射字典，键为变量不同取值，值为对应的连续化值（如某个类别取值的WOE）；
          若fit的X为DataFrame或二维数组，maps则为映射字典的字典，键为变量名，值为映射字典（同fit的X为Series时的maps值）。
    """
    __metaclass__=ABCMeta
    
    def __init__(self,feature_names=None,return_array=False):
        """
        feature_names：需要连续化的变量列表，非负整数或者字符串。
        return_array: True则统一返回numpy.ndarray，Fasle则返回形式同X。
        """
        self.feature_names=feature_names
        self.return_array=return_array
    
    @abstractmethod
    def fit(self,X,y):
        '''
        对feature_names中的变量获取各自连续化的映射字典。
        X：DataFrame或二维数组，或Series或一维数组。
        y：Series，index需要与X对应，或一维数组（暂时只支持0-1）。
        '''
        pass
    
    def transform(self,X):
        '''
        连续化数据：缺失值不作任何操作，对于不存在于已知类别的样本，会设置为缺失值。
        X：DataFrame或二维数组，或Series或一维数组。
        返回连续化后的数据，DataFrame或二维数组，或Series或一维数组。
        '''
        X=X.copy()
        if len(X.shape)==1:
            data=pd.Series(X)
            data=data.map(lambda xx: self.maps.get(xx,np.nan))
        else:
            data=pd.DataFrame(X)
            if isinstance(X,pd.DataFrame):
                for feature in self.maps:
                    data[feature]=data[feature].map(lambda xx: self.maps[feature].get(xx,np.nan))
            else:
                for feature in self.maps:
                    data.iloc[:,feature]=data.iloc[:,feature].map(lambda xx: self.maps[feature].get(xx,np.nan))
        if isinstance(X,np.ndarray) or self.return_array:
            return data.values
        else:
            return data

    def fit_transform(self,X,y):
        return self.fit(X,y).transform(X)
    
    def get_crosstab(self,X,y):
        '''
        对feature_names中的变量获取各自的列联表。
        X：DataFrame或二维数组，或Series或一维数组。
        y：Series，index需要与X对应（暂时只支持0-1），或一维数组。
        返回列联表DataFrame（X为Series或一维数组）或字典（X为DataFrame或二维数组，键为变量名，值为列联表DataFrame）
        '''
        if len(X.shape)==1:
            result=pd.crosstab(X,y)
        else:
            result={}
            if self.feature_names is None:
                if isinstance(X,pd.DataFrame):
                    feature_names=list(X.columns)
                else:
                    feature_names=[i for i in range(X.shape[1])]
            else:
                feature_names=self.feature_names
            if isinstance(X,pd.DataFrame):
                for feature in feature_names:
                    result[feature]=pd.crosstab(X[feature],y)
            else:
                for feature in feature_names:
                    result[feature]=pd.crosstab(X[:,feature],y)
        return result

    def plot(self,rot=0,pattern='\((.*?),',str_nopattern=None,
             close=True,show_last=True,figsize=(18,8),
             xlabel='Discrete Values',ylabel='Continuous Values',fontsize=12):
        '''
        可视化每个离散变量的连续化值（直方图），可以看离散变量在每个类别中的概率值或WOE分布情况。
        rot：文字旋转角度。
        pattern: 正则表达式，用于匹配横轴标签字符串，使其按照该正则表达式提取后的数值排序。
        str_nopattern: 若self.single为True，则为列表，表示未匹配pattern字符串的正常顺序；
                       若self.single为True，则为字典，键为变量名（或变量位置），值为列表，含义同上。
        返回字典：键为变量名（对于单变量，默认为空字符串''），值为图片对象。
        '''
        result={}
        if self.single is True:
            data=pd.Series(self.maps)
            data.index.name=''
            data.name=''
            data=data.reindex(utils.sort(data.index.tolist(),ascending=True,pattern=pattern,str_nopattern=str_nopattern,converter=float))
            fig=plt.figure(figsize=figsize)
            ax=fig.add_subplot(111)
            data.plot(kind='bar',rot=rot,ax=ax,fontsize=fontsize)
            ax.set_xlabel(xlabel,fontsize=fontsize)
            ax.set_ylabel(ylabel,fontsize=fontsize)
            ax.set_title(self.method,fontsize=fontsize)
            result['']=fig
            if close:
                plt.close('all')
        else:
            if str_nopattern is None:
                str_nopattern={}
            for feature in self.maps:
                data=pd.Series(self.maps[feature])
                data.index.name=''
                data.name=''
                data=data.reindex(utils.sort(data.index.tolist(),ascending=True,pattern=pattern,str_nopattern=str_nopattern.get(feature,None),converter=float))
                fig=plt.figure(figsize=figsize)
                ax=fig.add_subplot(111)
                data.plot(kind='bar',rot=rot,ax=ax,fontsize=fontsize)
                ax.set_xlabel(xlabel, fontsize=fontsize)
                ax.set_ylabel(ylabel, fontsize=fontsize)
                ax.set_title('%s : %s'%(self.method,feature),fontsize=fontsize)
                result[feature]=fig
                if close:
                    plt.close('all')
        if show_last:
            try:
                fig
                fig.show()
            except:
                pass
        return result

class ProbContinuous(BaseContinuous):
    def __init__(self,feature_names=None,return_array=False):
        '''
        条件概率连续化：p=P(y=1)，类别c对应的连续化取值为(N(x=c,y=1)+p)/(N(x=c)+1)。
        feature_names：需要连续化的变量列表，非负整数或者字符串。
        return_array: True则统一返回numpy.ndarray，Fasle则返回形式同X。
        '''
        BaseContinuous.__init__(self,feature_names=feature_names,return_array=return_array)
        self.method='Prob'
    
    @staticmethod
    def cal_prob(crosstab):
        '''
        根据列联表计算条件概率映射表，类别c对应的连续化取值为(N(x=c,y=1)+p)/(N(x=c)+1)。
        crosstab：列联表DataFrame（index为变量取值类别，column为y的标签0/1）。
        返回映射字典，键为取值类别，值为对应的条件概率值。
        '''
        total=crosstab.sum(axis=0)
        p=total.loc[1]/total.sum()
        N=crosstab.sum(axis=1)+1
        N1=crosstab[1]+p
        N.name=''
        N.index.name=''
        N1.name=''
        N1.index.name=''
        return dict(N1/N)
        
    def fit(self,X,y):
        self.crosstabs=self.get_crosstab(X,y)
        if len(X.shape)==1:
            self.maps=self.cal_prob(self.crosstabs)
            self.single=True
        else:
            self.single=False
            self.maps={}
            for feature in self.crosstabs:
                self.maps[feature]=self.cal_prob(self.crosstabs[feature])
        return self


class WoeContinuous(BaseContinuous):
    def __init__(self,feature_names=None,return_array=False):
        '''
        WOE连续化。
        feature_names：需要连续化的变量列表，非负整数或者字符串。
        return_array: True则统一返回numpy.ndarray，Fasle则返回形式同X。
        '''
        BaseContinuous.__init__(self,feature_names=feature_names,return_array=return_array)
        self.method='WOE'
    
    @staticmethod
    def cal_woe(crosstab):
        '''
        根据列联表计算WOE映射表，类别c对应的WOE定义为log(r(x=c,y=1)/r(x=c,y=0))，
        其中r(x=c,y=1)=N(x=c,y=1)/N(y=1)为坏人占比，r(x=c,y=0)=N(x=c,y=0)/N(y=0)为好人占比。
        crosstab：列联表DataFrame（index为变量取值类别，column为y的标签0/1）。
        返回映射字典，键为取值类别，值为对应的WOE。
        '''
        tmp=crosstab.copy()
        #先对列联表微调，防止出现无穷
        tmp[tmp==0]=1
        r=tmp/tmp.sum(axis=0)
        result=np.log(r[1]/r[0])
        return dict(result)
        
    def fit(self,X,y):
        self.crosstabs=self.get_crosstab(X,y)
        if len(X.shape)==1:
            self.maps=self.cal_woe(self.crosstabs)
            self.single=True
        else:
            self.single=False
            self.maps={}
            for feature in self.crosstabs:
                self.maps[feature]=self.cal_woe(self.crosstabs[feature])
        return self
    
    def cal_iv(self):
        '''
        计算变量的IV值（须先fit）。
        返回数值（单变量）或IV映射字典，视fit传入的X而定。
        '''
        if not hasattr(self,'crosstabs'):
            raise Exception('须先fit才能计算IV值！')
        crosstabs=copy.deepcopy(self.crosstabs)
        if not isinstance(crosstabs,dict):
            crosstabs[crosstabs==0]=1
            r=crosstabs/crosstabs.sum(axis=0)
            result=((r[1]-r[0])*np.log(r[1]/r[0])).sum()
        else:
            result={}
            for feature in crosstabs:
                crosstabs[feature][crosstabs[feature]==0]=1
                r=crosstabs[feature]/crosstabs[feature].sum(axis=0)
                result[feature]=((r[1]-r[0])*np.log(r[1]/r[0])).sum()
        
        return result

    def plot_iv(self,top=5,rot=0,fontsize=12,xlabel='Feature Name',ylabel='Information Value',
                title='Information Value of All Features',figsize=(18,8),close=True):
        '''
        计算变量的IV值并排序，画出直方图。
        top：画图筛选排名前top，若为负值或0，则不筛选。
        rot：文字旋转角度。
        close：是否关闭图片。
        返回二元组，第一项为所有变量IV值的dataframe（shape为[变量数,1]，index为变量名），第二项为直方图对象。
        '''
        IVs=self.cal_iv()
        if self.single:
            IVs=pd.DataFrame([IVs])
            IVs.index=['']
            IVs.columns=['IV']
        else:
            IVs=pd.Series(IVs)
            IVs.name='IV'
            IVs=IVs.sort_values(ascending=False)
            IVs=pd.DataFrame(IVs)
        fig=plt.figure(figsize=figsize)
        ax=fig.add_subplot(111)
        if top<=0:
            IVs.plot(kind='bar',rot=rot,legend=False,ax=ax,fontsize=fontsize)
        else:
            IVs.head(top).plot(kind='bar',rot=rot,legend=False,ax=ax,fontsize=fontsize)
        ax.set_xlabel(xlabel,fontsize=fontsize)
        ax.set_ylabel(ylabel,fontsize=fontsize)
        ax.set_title(title,fontsize=fontsize)
        if close:
            plt.close('all')
        return IVs,fig


class LabelContinuous(BaseEstimator,TransformerMixin):
    '''
    改写的LabelEncoder，主要用于将名义变量的取值转换为0到n_var-1，其中n_var为变量var可能的取值总数。
    同时将未知类别当作缺失值处理（sklearn.preprocessing.LabelEncoder没有这功能）。
    
    属性：
    feature_names：需要转换的变量列表，非负整数或者字符串。
    return_array: True则统一返回numpy.ndarray，Fasle则返回形式同X。
    maps：字典，键为变量名称，值为映射字典（其键为原始编码，值为新的数字编码）。
    maps_inverse：字典，键为变量名称，值为逆向映射字典（其键为新的数字编码，值为原始编码）。
    '''
    def __init__(self,feature_names=None,return_array=False):
        self.feature_names=feature_names
        self.return_array=return_array
    
    def fit(self,X,y=None):
        '''
        对feature_names中的变量获取各自的映射字典。
        X：DataFrame或二维数组。
        y：不需要提供。
        '''
        if self.feature_names is None:
            if isinstance(X,pd.DataFrame):
                feature_names=list(X.columns)
            else:
                feature_names=[i for i in range(X.shape[1])]
        else:
            feature_names=self.feature_names
        if isinstance(X,pd.DataFrame):
            df=True
        else:
            df=False
        maps={}
        maps_inverse={}
        if not df:
            X=pd.DataFrame(X)
            X.columns=[i for i in range(X.shape[1])]
        for feature in feature_names:
            unique=X.loc[X[feature].notnull(),feature].unique()
            unique=unique.tolist()
            unique.sort()
            if len(unique)==0:
                raise Exception('变量%s全缺失！'%feature)
            result_tmp={value:i for i,value in enumerate(unique)}
            result_tmp_inverse={i:value for i,value in enumerate(unique)}
            maps[feature]=result_tmp.copy()
            maps_inverse[feature]=result_tmp_inverse.copy()
        self.maps=copy.deepcopy(maps)
        self.maps_inverse=copy.deepcopy(maps_inverse)
        return self
    
    def transform(self,X):
        '''
        将X转换为新的编码，未知类别转换为缺失值。
        X：DataFrame或二维数组。
        返回转换后的DataFrame（形式与X相同）或二维数组。
        '''
        data=pd.DataFrame(X.copy())
        for feature in self.maps:
            data[feature]=data[feature].map(lambda xx: self.maps[feature].get(xx,np.nan))
        if isinstance(X,np.ndarray) or self.return_array:
            return data.values
        else:
            return data
    
    def inverse_transform(self,X):
        '''
        将X转换为原来的编码，缺失值不变。
        X：DataFrame或二维数组。
        返回转换后的DataFrame（形式与X相同）或二维数组。
        '''
        data = pd.DataFrame(X.copy())
        for feature in self.maps:
            data[feature] = data[feature].map(lambda xx: self.maps_inverse[feature].get(xx, np.nan))
        if isinstance(X, np.ndarray) or self.return_array:
            return data.values
        else:
            return data


class OneHotContinuous(BaseEstimator,TransformerMixin):
    '''
    改写的独热编码，主要用于将离散变量进行独热编码。若离散变量为字符型或者取值范围不是0到n_var-1，则需要先使用LabelContinuous做一步转换。
    同时将未知类别当作缺失值处理，独热编码的结果全为0。
    
    属性：
    feature_names：需要独热化的变量列表，默认为所有变量。
    feature_tolabel：需要使用LabelContinuous进行编码的字符型变量列表，默认为所有变量。
    drop_others：最终结果是否删除其他变量，True时若X为二维数组，则会将未独热化的列按顺序追加到后面。
    sep：独热编码变量的名称分隔符，如'x=a'，'y:b'。
    handle_na：非负整数，处理缺失值时填补使用，对最终结果不会造成影响。
    return_array: True则统一返回numpy.ndarray，Fasle则返回形式同X。
    kwds：参见sklearn.preprocessing.OneHotEncoder的相关参数。

    label_continuous_：LabelContinuous对象。
    one_hot_enc_：sklearn.preprocessing.OneHotEncoder对象。
    feature_maps：变量映射列表，例如[('x1',0),('x1',1),('x2','a'),('x2','b')]，表示结果数组的第0列为变量x1等于0，第3列为变量x2等于b。
    feature_map_names：变量映射名称列表，例如上面的例子对应的feature_map_names可能为['x1=0','x1=1','x2=a','x2=b']。
    '''
    
    def __init__(self,feature_names=None,feature_tolabel=None,drop_others=False,sep='=',handle_na=99,return_array=False,**kwds):
        self.feature_names=feature_names
        self.feature_tolabel=feature_tolabel
        self.drop_others=drop_others
        self.sep=sep
        self.handle_na=handle_na
        self.return_array=return_array
        self.kwds=kwds
        if feature_names is None:
            self.all_onehot=True
        else:
            self.all_onehot=False
        if feature_tolabel is None:
            self.all_tolabel=True
        else:
            self.all_tolabel=False
    
    def fit(self,X,y=None):
        '''
        对feature_names中的变量获取独热编码所需数据。
        X：DataFrame或二维数组。
        y：不需要提供。
        '''
        df=isinstance(X,pd.DataFrame)
        if self.all_onehot:
            if df:
                self.feature_names=list(X.columns)
            else:
                self.feature_names=[i for i in range(X.shape[1])]
        if self.all_tolabel:
            self.feature_tolabel=self.feature_names
        else:
            self.feature_tolabel=[col for col in self.feature_tolabel if col in self.feature_names]
        self.label_continuous_=LabelContinuous(feature_names=self.feature_tolabel,return_array=False)
        Xnew=self.label_continuous_.fit_transform(X)

        Xnew=pd.DataFrame(Xnew)
        
        #需要先填补缺失，否则会出错，训练模型而言缺失填补方法不会造成任何影响
        Xnew[self.feature_names]=Xnew[self.feature_names].fillna(method='ffill')
        Xnew[self.feature_names]=Xnew[self.feature_names].fillna(method='bfill')
        if Xnew[self.feature_names].isnull().sum().sum()>0:
            raise Exception('存在某一列全部缺失！')
        self.one_hot_enc_=OneHotEncoder(sparse=False,handle_unknown='ignore',**self.kwds)
        self.one_hot_enc_.fit(Xnew[self.feature_names])
        
        self.feature_maps=[]
        #根据sklearn.preprocessing.OneHotEncoder对象的属性获取每一列的变量数字编码含义
        if self.one_hot_enc_.n_values!='auto':
            for i,feature in enumerate(self.feature_names):
                for j in range(self.one_hot_enc_.feature_indices_[i+1]-self.one_hot_enc_.feature_indices_[i]):
                    self.feature_maps.append((feature,j))
        else:
            active_features_=self.one_hot_enc_.active_features_.tolist()
            for i,feature in enumerate(self.feature_names):
                for j in range(self.one_hot_enc_.feature_indices_[i+1]-self.one_hot_enc_.feature_indices_[i]):
                    if j+self.one_hot_enc_.feature_indices_[i] in active_features_:
                        self.feature_maps.append((feature,j))
        #根据LabelContinuous对象的属性获取每一列的变量真实编码含义
        def f(xx):
            feature=xx[0]
            value=xx[1]
            if feature in self.feature_tolabel:
                return (feature,self.label_continuous_.maps_inverse[feature][value])
            else:
                return (feature,value)
        self.feature_maps=[f(xx) for xx in self.feature_maps]
        
        def fmt(xx,sep):
            feature=xx[0]
            value=xx[1]
            try:
                return '%s%s%g'%(feature,sep,value)
            except:
                return '%s%s%s'%(feature,sep,value)
        
        self.feature_map_names=[fmt(xx,self.sep) for xx in self.feature_maps]
        return self
    
    def transform(self,X):
        '''
        将X转换为独热编码，未知类别（包括缺失值）转换为全为0。
        X：DataFrame或二维数组。
        返回转换后的DataFrame或二维数组，前几列为独热编码变量，其他的变量按列的顺序合并到后面（如果self.drop_others为False）。
        '''
        df=isinstance(X,pd.DataFrame)
        if df:
            X_others=X[[col for col in X.columns.tolist() if col not in self.feature_names]].copy()
        else:
            X_others=X[:,[col for col in range(X.shape[1]) if col not in self.feature_names]].copy()
            X_others=pd.DataFrame(X_others)
        Xnew=self.label_continuous_.transform(X)

        Xnew=pd.DataFrame(Xnew)
        Xnew[self.feature_names]=Xnew[self.feature_names].fillna(self.handle_na)
        Xnew=self.one_hot_enc_.transform(Xnew[self.feature_names])
        Xnew=pd.DataFrame(Xnew)
        Xnew.columns=self.feature_map_names
        if df:
            Xnew.index=X.index
        Xnew=Xnew.astype(int)
        if self.drop_others:
            result=Xnew
        else:
            result=pd.concat([Xnew,X_others],axis=1)
        if (not df) or self.return_array:
            return result.values
        else:
            result.index.name=X.index.name
            return result


#==============================================================================
# 测试
#==============================================================================

def test():
    #构建测试数据
    np.random.seed(13)
    X=pd.DataFrame(np.random.randn(20,2),columns=['cont1','cont2'])
    X['cate1']=np.random.choice([1,2,3,4,5],20)
    X['cate2']=np.random.choice([1,2,3,4,5],20)
    y=pd.Series(np.random.choice([0,1],20))
    
    #条件概率连续化
    clf=ProbContinuous(feature_names=['cate1','cate2'])
    clf.fit(X,y)
    clf.plot(rot=0,close=False)
    result1=clf.transform(X)
    result2=clf.fit_transform(X,y)
    
    crosstabs=clf.crosstabs
    maps=clf.maps
    
    #WOE连续化（同时可计算IV值）
    clf=WoeContinuous(feature_names=['cate1','cate2'])
    clf.fit(X,y)
    clf.plot_iv(top=5,fontsize=18,close=False)
    result1=clf.transform(X)
    result2=clf.fit_transform(X,y)
    
    crosstabs=clf.crosstabs
    maps=clf.maps
    IVs=clf.cal_iv()
    
    #标签映射
    X_train=pd.DataFrame(np.random.choice(['a','b','c','d'],[10,3]),columns=['x1','x2','x3'])
    X_test=pd.DataFrame(np.random.choice(['a','b','c','d'],[5,3]),columns=['x1','x2','x3'])
    X_test[X_test=='d']='f'
    
    clf=LabelContinuous()
    X_train_new=clf.fit_transform(X_train)
    X_test_new=clf.transform(X_test)
    X_test_inverse=clf.inverse_transform(X_test_new)
    
    #改写的独热编码
    X_train=pd.DataFrame(np.random.choice(['a','b','c','d'],[10,3]),columns=['x1','x2','x3'])
    X_test=pd.DataFrame(np.random.choice(['a','b','c','d'],[5,3]),columns=['x1','x2','x3'])
    X_test[X_test=='d']='f'
    
    clf=OneHotContinuous(feature_names=['x1','x2','x3'])
    X_train_new=clf.fit_transform(X_train)
    X_test_new=clf.transform(X_test)
    




