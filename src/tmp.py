# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
import sklearn.preprocessing
import sklearn.decomposition
import discretize
import continuous
import copy

cols_input_cont=(discretize.BaseDiscretizer,)
cols_input_cate=(continuous.BaseContinuous,)
cols_input_cate_mult=(continuous.OneHotContinuous,)

cols_output_userdefine=(sklearn.decomposition.PCA,)

add_feature_cate_two=(continuous.OneHotContinuous,)
add_feature_cate_mult=(discretize.BaseDiscretizer,)

class OneStepTransformer(BaseEstimator,TransformerMixin):
    '''
    针对自定义数据结构BasicDataStruct实现单步的transformer。
    初始化输入：
        name：该步转换的名称，字符串。
        transformer：转换实例，如sklearn.preprocessing.StandardScaler()。
        params：参数字典，用于自定义transformer的输入输出处理方式。参数如下：
                cols_input=None输入变量列表，默认会根据transformer类别的不同而不同；
                cols_output=None输出变量列表，默认会根据transformer类别的不同而不同；
                drop_others=False是否删除其余变量，True则会新建BasicDataStruct对象（仅保留输出变量），False则会使用BasicDataStruct.add；
                feature_cate_two=None参见BasicDataStruct.add参数含义，默认会根据transformer类别的不同而不同；
                feature_cate_mult=None参见BasicDataStruct.add参数含义，默认会根据transformer类别的不同而不同；
                replace=True参见BasicDataStruct.add参数含义；
                ignore=False参见BasicDataStruct.add参数含义；
                prefix=''参见BasicDataStruct.add参数含义；
                suffix=None参见BasicDataStruct.add参数含义，默认为转换名称name；
    
    
    '''
    def __init__(self,name,transformer,params):
        self.name=name
        self.transformer=transformer
        self.cols_input=params.get('cols_input',None)
        self.cols_output=params.get('cols_output',None)
        self.drop_others=params.get('drop_others',False)
        self.feature_cate_two=params.get('feature_cate_two',None)
        self.feature_cate_mult=params.get('feature_cate_mult',None)
        self.replace=params.get('replace',True)
        self.ignore=params.get('ignore',False)
        self.prefix=params.get('prefix','')
        self.suffix=params.get('suffix',name)
        
    def fit(self,data_model):
        '''
        data_model：BasicDataStruct数据类型。
        '''
        if self.cols_input is None:
            if isinstance(self.transformer,cols_input_cont):
                self.cols_input=data_model.feature_cont
            elif isinstance(self.transformer,cols_input_cate_mult):
                self.cols_input=data_model.feature_cate_mult
            elif isinstance(self.transformer,cols_input_cate):
                self.cols_input=data_model.feature_cate_mult+data_model.feature_cate_two
            else:
                self.cols_input=data_model.X.columns.tolist()
        elif isinstance(self.cols_input,str):
            if self.cols_input=='cont':
                self.cols_input=data_model.feature_cont
            elif self.cols_input=='cate_mult':
                self.cols_input=data_model.feature_cate_mult
            elif self.cols_input=='cate':
                self.cols_input=data_model.feature_cate_mult+data_model.feature_cate_two
            else:
                self.cols_input=data_model.X.columns.tolist()
        
        if hasattr(self.transformer,'fit_transform'):
            Xnew=self.transformer.fit_transform(data_model.X[self.cols_input],data_model.Y)
        else:
            Xnew=self.transformer.fit(data_model.X[self.cols_input],data_model.Y).transform(data_model.X[self.cols_input])
        
        if self.cols_output is None:
            if isinstance(Xnew,pd.DataFrame):
                self.cols_output=Xnew.columns.tolist()
            elif hasattr(self.transformer,'get_feature_names'):
                self.cols_output=self.transformer.get_feature_names()
            elif isinstance(self.transformer,cols_output_userdefine):
                self.cols_output=[self.name+'_'+str(i) for i in range(Xnew.shape[1])]
            else:
                self.cols_output=self.cols_input
        elif isinstance(self.cols_output,str):
            if self.cols_output=='result':
                self.cols_output=Xnew.columns.tolist()
            elif self.cols_output=='userdefine':
                self.cols_output=[self.name+'_'+str(i) for i in range(Xnew.shape[1])]
            else:
                self.cols_output=self.cols_input
        
        if self.feature_cate_two is None:
            if isinstance(self.transformer,add_feature_cate_two):
                self.feature_cate_two=self.cols_output
            else:
                self.feature_cate_two=[]
        if self.feature_cate_mult is None:
            if isinstance(self.transformer,add_feature_cate_mult):
                self.feature_cate_mult=self.cols_output
            else:
                self.feature_cate_mult=[]
        
        self.feature_cate_two.extend([col for col in self.cols_output if (col not in self.feature_cate_two) and (col in data_model.feature_cate_two)])
        self.feature_cate_mult.extend([col for col in self.cols_output if (col not in self.feature_cate_mult) and (col in data_model.feature_cate_mult)])
        
        return self
    
    def transform(self,data_model):
        '''
        data_model：BasicDataStruct数据类型。
        '''
        newdata_model=copy.deepcopy(data_model)
        Xnew=self.transformer.transform(newdata_model.X[self.cols_input])
        


