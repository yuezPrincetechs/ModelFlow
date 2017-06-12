# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import src.utils.utils as utils
import copy


class BasicDataStruct():
    '''
    基本的数据结构，方便数据预处理、特征提取等各类操作。
    定义一个数据类，拥有以下几个属性：
        X：dataframe，有列名（变量名）和索引（每个样本的ID）。
        Y：series，有索引（对应X的索引，标识每个样本）。
        feature_cont：连续型变量列表。
        feature_binary：0-1离散型变量列表（不需要one hot，可以直接使用）。
        feature_cate：多取值离散型变量列表（需要one hot或者连续化）。
        model_type：标识分类或者回归。
    
    同时拥有以下几种方法（操作），可能会对变量列表进行增、删等：
        delete：删除已有的列，并同步修改变量列表。
        select：选择已有的某几列，并同步修改变量列表，实际上和delete方法的作用差不多，只是方便选择。
        add：增加新的列，并同步修改变量列表（对已有列可以考虑替换或者重命名后增加）。
        addObj：将当前对象与另一个对象合并。
        delFeature：改变离散型变量列表（删）。
        addFeature：改变离散型变量列表（增）。
        rename：改变变量名称。
        current_state：打印当前状态，包括当前各类变量的数量及名称。

    '''
    
    def __init__(self,X,Y=None,feature_binary=None,feature_cate=None,model_type='classification'):
        '''
        X：dataframe，有列名（变量名）和索引（每个样本的ID）；或者series，有name属性和索引。。
        Y：series，有索引（对应X的索引，标识每个样本），默认可以不提供。
        feature_binary：0-1离散型变量列表（不需要one hot，可以直接使用），默认为空。
        feature_cate：多取值离散型变量列表（需要one hot或者连续化），默认为空。
        model_type：标识分类或者回归，可选为{'regression','classification'}，默认为'classification'。
        '''
        if feature_binary is None:
            feature_binary=[]
        if feature_cate is None:
            feature_cate=[]
        if len(X.shape)==2:
            self.X=X.copy()
        else:
            self.X=pd.DataFrame(X)
        if Y is None:
            self.Y=None
        else:
            self.Y=Y.copy()
        self.feature_binary=[col for col in self.X.columns if col in feature_binary]
        self.feature_cate=[col for col in self.X.columns if col in feature_cate]
        self.feature_cont=[col for col in self.X.columns if col not in feature_binary+feature_cate]
        self.model_type=model_type
    
    def delete(self,features_todel,errors='ignore'):
        '''
        删除已有列，并同步修改变量列表。
        features_todel：需要删除的变量列表。
        errors：{'ignore','raise'}，默认为'ignore'。
        '''
        self.X.drop(features_todel,axis=1,inplace=True,errors=errors)
        self.delFeature(features_todel=features_todel)
    
    def select(self,features_selected):
        '''
        选择已有的某几列，并同步修改变量列表。
        features_selected：需要选择的变量列表。
        '''
        features_todel=[col for col in self.X.columns.tolist() if col not in features_selected]
        self.X.drop(features_todel,axis=1,inplace=True,errors='ignore')
        self.delFeature(features_todel=features_todel)
    
    def add(self,data,feature_binary=None,feature_cate=None,replace=True,ignore=False,prefix='',suffix='_new'):
        '''
        增加新的列，并同步修改变量列表（对已有列可以考虑替换或者忽略或者重命名后增加）。
        data：dataframe，有列名（变量名）和索引（每个样本的ID）；或者series，有name属性和索引。
        feature_binary：0-1离散型变量列表，默认为空。
        feature_cate：多取值离散型变量列表，默认为空。
        replace：是否替换已有列。若不替换已有列，且ignore为False时，需要重命名新列（根据prefix和suffix）。
        ignore：是否忽略已有列，仅在replace为False的情况下有效。
        prefix：已有列名需要增加的前缀。
        suffix：已有列名需要增加的后缀。
        '''
        obj=BasicDataStruct(X=data,Y=None,
                            feature_binary=feature_binary,
                            feature_cate=feature_cate,
                            model_type=self.model_type)
        self.addObj(obj,replace=replace,ignore=ignore,prefix=prefix,suffix=suffix)
        
        
    def addObj(self,obj,replace=True,ignore=False,prefix='',suffix='_new'):
        '''
        将BasicDataStruct对象obj合并到self中，类似于增加新列、替换已有列（视replace和ignore的值而定）。
        obj：BasicDataStruct对象实例。
        replace：是否替换已有列。若不替换已有列，且ignore为False时，需要重命名新列（根据prefix和suffix）。
        ignore：是否忽略已有列，仅在replace为False的情况下有效。
        prefix：已有列名需要增加的前缀。
        suffix：已有列名需要增加的后缀。
        '''
        newobj=copy.deepcopy(obj)
        features_exist=list(set(self.X.columns)&set(newobj.X.columns))
        if replace is True:
            #如果替换，则先删除self的已有列（self与newobj共有列），保证self与newobj的列没有交集
            self.delete(features_todel=features_exist)
        elif ignore:
            #如果不替换且需要忽略，则先删除newobj的已有列（self与newobj共有列），保证self与newobj的列没有交集
            newobj.delete(features_todel=features_exist)
        else:
            #如果新增，则先重命名newobj中与self的共有列，同样保证self与newobj的列没有交集
            rename_rule={col:prefix+col+suffix for col in features_exist}
            newobj.rename(rename_rule)
        self.X=pd.concat([self.X,newobj.X],axis=1)
        #更新离散型变量列表
        self.addFeature(feature_binary=newobj.feature_binary,feature_cate=newobj.feature_cate)
    
    def delFeature(self,features_todel=None):
        '''
        仅从离散型变量列表中删除变量，不影响具体的数据，被删除变量默认转变为连续型。
        features_todel：需要删除的变量列表（仅对离散型变量列表中的变量有效）。
        '''
        if features_todel is None:
            features_todel=[]
        self.feature_binary=utils.delFromList(self.feature_binary,features_todel)
        self.feature_cate=utils.delFromList(self.feature_cate,features_todel)
        self.feature_cont=[col for col in self.X.columns if col not in self.feature_binary+self.feature_cate]
    
    def addFeature(self,feature_binary=None,feature_cate=None):
        '''
        仅在离散型变量列表中增加变量，不影响具体的数据，被增加变量会从连续型变量列表中剔除。
        feature_binary：需要增加的0-1离散型变量列表，默认为空。
        feature_cate：需要增加的多取值离散型变量列表，默认为空。
        '''
        if feature_binary is None:
            feature_binary=[]
        if feature_cate is None:
            feature_cate=[]
        self.feature_binary=utils.addToList(self.feature_binary,feature_binary)
        self.feature_cate=utils.addToList(self.feature_cate,feature_cate)
        self.feature_cont=[col for col in self.X.columns if col not in self.feature_binary+self.feature_cate]

    def rename(self,rename_rule=None):
        '''
        重命名变量名。
        rename_rule：重命名规则字典，键为原始变量名，值为新变量名，默认None不进行任何操作。
        '''
        if rename_rule is None:
            return None
        self.X=self.X.rename(columns=rename_rule)
        self.feature_binary=[rename_rule.get(col,col) for col in self.feature_binary]
        self.feature_cate=[rename_rule.get(col,col) for col in self.feature_cate]
    
    def current_state(self):
        '''
        获取当前状态：模型类型、数据大小及连续型、离散型变量的数量及名称。
        返回字符串。
        '''
        feature_binary=self.feature_binary
        feature_cate=self.feature_cate
        feature_cont=self.feature_cont
        
        content='*'*60+'\n'
        content+='model type: %s\n\n'%self.model_type
        content+='shape of X: (%s , %s)\n\n'%(self.X.shape[0],self.X.shape[1])
        content+='features of continuous (%s): %s\n\n'%(len(feature_cont),', '.join(feature_cont))
        content+='features of binary (%s): %s\n\n'%(len(feature_binary),', '.join(feature_binary))
        content+='features of categorical (%s): %s\n\n'%(len(feature_cate),', '.join(feature_cate))
        content+='*'*60
        return content


#==============================================================================
# 测试
#==============================================================================
def test():
    #构建测试数据
    np.random.seed(13)
    X=pd.DataFrame(np.random.randn(10,4),columns=['cont1','cont2','cont3','cont4'])
    X['cate_two1']=np.random.choice([0,1],10)
    X['cate_two2']=np.random.choice([0,1],10)
    X['cate_mult1']=np.random.choice([1,2,3,4,5],10)
    X['cate_mult2']=np.random.choice([1,2,3,4,5],10)
    
    feature_cate_two=['cate_two1','cate_two2']
    feature_cate_mult=['cate_mult1','cate_mult2']
    
    #创建对象实例
    model=BasicDataStruct(X,Y=None,
                          feature_cate_two=feature_cate_two,
                          feature_cate_mult=feature_cate_mult,
                          model_type='classification')
    print(model.current_state())
    
    #通过变换构造新变量，替换或者新增
    #新增
    x_exp=np.exp(model.X['cont1'])
    model.add(x_exp,replace=False,ignore=False,suffix='_exp')
    print(model.current_state())
    
    #替换
    model.add(np.sin(model.X[['cont2','cont3']]),replace=True)
    print(model.current_state())
    
    #连续变量离散化后新增
    import src.discretize as discretize
    clf=discretize.QuantileDiscretizer()
    model.add(clf.fit_transform(model.X['cont4']),
              feature_cate_mult=['cont4'],replace=False,ignore=False,suffix='dis')
    print(model.current_state())
    
    #连续变量离散化后替换
    clf=discretize.QuantileDiscretizer()
    model.add(clf.fit_transform(model.X[['cont1']]),
              feature_cate_mult=['cont1'],replace=True)
    print(model.current_state())
    
    #删除已有变量
    model.delete(features_todel=['cont1_exp','cate_mult1'])
    print(model.current_state())
    
    #重命名
    model.rename({'cont4dis':'cont4_dis'})
    print(model.current_state())
    
    #改变离散型变量列表（增）
    model.addFeature(feature_cate_two=['cate_two1'],feature_cate_mult=['cont1'])
    print(model.current_state())
    
    #改变离散型变量列表（删）
    model.delFeature(feature_cate_two=['cate_two1'],feature_cate_mult=['cont1'])
    print(model.current_state())
    
    
    
    