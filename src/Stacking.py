# -*- coding: utf-8 -*-
'''
集成模型：stacking模块。

'''

import numpy as np
import pandas as pd
import copy
from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator,TransformerMixin,clone
from sklearn.model_selection import KFold,cross_val_score
import gc


class StackingTransformer(BaseEstimator,TransformerMixin):
    '''
    stacking模型转换类，主要解决如下问题：
    假设有两个stage，[[('s11',clf11),('s12',clf12)],[('s21',clf21),('s22',clf22),('s23',clf23),('s24',clf24)]]，
    stage1有两个模型，stage2有四个模型，需要使用stacking的方法训练模型和转换数据（不同于stacking的是，此转换类只是截取转换的
    那一部分，而非模型预测，目的是利用stacking生成新特征，方便与其他特征的融合），对于每一个模型，均会将数据按照交叉验证的形式
    训练出n_folds个模型，然后利用这n_folds个模型分别对数据做转换，最后做平均作为这n_folds个模型的转换结果。

    参数：
    stages: 列表，形如[[('s11',clf11),('s12',clf12)],[('s21',clf21),('s22',clf22),('s23',clf23),('s24',clf24)]]，
            其中每个元素也是列表（注意：元素个数需大于1），对应每个stage，如[('s11',clf11),('s12',clf12)]，其中模型任意
            （与sklearn兼容，有fit、predict_proba或predict即可）。
    type: {'classification','regression'}。
    n_folds: int，交叉验证的折数。
    return_array: True则统一返回numpy.ndarray，Fasle则返回形式同X。
    verbose: 是否打印日志信息。
    kwds: 字典，形式同pipeline的参数，如{'s11__max_depth':4,'s24__C':100}。
          注意：不同stage之间以及同一stage中的模型名称不能重复。

    属性：
    stages_: 列表，形式同stages，不同的是存储的均为训练好的模型，由于是根据交叉验证的形式训练模型，每一个模型都会对应n_folds个
             训练好的模型，即stages中的('s11',clf11)会被替换为形如('s11',[clf11_1,clf11_2,clf11_3])。
             注意：若某一模型（如clf12）本身就属于stacking系列，由于内部已经属于交叉验证的训练模式，此处训练则不需要交叉验证，即训练好的模型
             仍然只有一个，而不是列表（目的是避免重复交叉验证，模型数量和训练时间呈平方量级增长），如stages中的('s12',clf12)会仍被替换为形如
             ('s12',clf12_fit)。
    n_classes_: int，类的个数，仅当self.type='classification'时有该属性。


    '''

    def __init__(self,stages,type='classification',n_folds=5,return_array=False,verbose=False,**kwds):
        self.stages=stages
        self.type=type
        self.n_folds=n_folds
        self.return_array=return_array
        self.verbose=verbose
        self.kwds=copy.deepcopy(kwds)

    def fit(self,X,y):
        '''
        训练模型，若某一子模型属于stacking系列，则会特殊处理，否则均按照数据集划分、交叉验证的形式训练出多个模型。
        :param X: dataframe或二维数组。
        :param y: series（index需要与X对应）或一维数组。
        :return: self。
        '''
        n=X.shape[0]
        if isinstance(X,pd.DataFrame):
            X=X.values
        if isinstance(y,pd.Series):
            y=y.values
        ks=KFold(self.n_folds,shuffle=False)
        if self.type=='classification':
            self.n_classes_=len(set(y))
        self.stages_=[]
        for i,stage in enumerate(self.stages):
            if self.verbose:
                print('stage=%d 训练开始[---------->]'%i)
            stage_models=[]
            stage_X=[]
            for name,clf in stage:
                stage_models_cv=[]
                if isinstance(clf,(StackingClassifier,StackingRegressor)):
                    clf=clone(clf)
                    params={k.replace('%s__'%name,''):self.kwds[k] for k in self.kwds if k.startswith('%s__'%name)}
                    params['n_folds']=self.n_folds
                    clf.set_params(**params)
                    clf.fit(X,y)
                    stage_models.append((name,clf))
                    if i<len(self.stages)-1:
                        # 注意分类和回归在fit阶段的不同处理方式
                        if self.type=='classification':
                            output_tmp=clf.predict_proba(X,train=True)[:,1:].reshape((n,-1))
                            stage_X.append(output_tmp.copy())
                        elif self.type=='regression':
                            output_tmp = clf.predict(X,train=True).reshape((n, -1))
                            stage_X.append(output_tmp.copy())
                        del output_tmp
                        gc.collect()
                else:
                    params={k.replace('%s__'%name,''):self.kwds[k] for k in self.kwds if k.startswith('%s__' % name)}
                    output=None
                    for j,(train_index,test_index) in enumerate(ks.split(X)):
                        newclf=clone(clf)
                        newclf.set_params(**params)
                        newclf.fit(X[train_index],y[train_index])
                        stage_models_cv.append(newclf)
                        if i < len(self.stages) - 1:
                            if self.type=='classification':
                                output_tmp=newclf.predict_proba(X[test_index])[:,1:].reshape((test_index.shape[0],-1))
                            elif self.type=='regression':
                                output_tmp=newclf.predict(X[test_index]).reshape((test_index.shape[0],-1))
                            if j==0:
                                output=np.full((n,output_tmp.shape[1]),np.nan)
                            output[test_index,:]=output_tmp.copy()
                            del output_tmp
                            gc.collect()
                    if i<len(self.stages)-1:
                        stage_X.append(output.copy())
                        del output
                        gc.collect()
                    stage_models.append((name,stage_models_cv.copy()))
                if self.verbose:
                    print('stage=%d model=%s 训练完成'%(i,name))
            self.stages_.append(stage_models.copy())
            if i<len(self.stages)-1:
                X=np.concatenate(stage_X,axis=1)
            del stage_X
            gc.collect()
            if self.verbose:
                print('stage=%d 训练完成[<----------]\n'%i)
        return self

    def transform(self,X,train=False):
        '''
        转换数据，使用每个stage内的每个子模型对上层数据进行预测并输出，作为下个stage的输入。
        若经过交叉验证得到n_folds个模型：
            当train=False时，会同时对整个数据集做预测，然后求平均作为下个stage的输入；
            当train=True时，属于训练过程时的转换，先将数据集分为几折，然后分别用对应的模型预测得到对应的输出，而非全部预测取平均。
        :param X: dataframe或二维数组。
        :param train: 布尔值，是否为训练阶段，也即输入的X是否为fit阶段的X，训练阶段使用的X和测试阶段使用的X的转换方式应该有所不同，
                      如果均使用求平均的转换方式，训练阶段使用的X得到的转换概率存在过拟合嫌疑。
        :return: 若self.return_array为True，则返回二维数组，列按照self.stages_的最后一层模型输出的顺序合并；
                 若self.return_array为False，则返回形式同X，列按照self.stages_的最后一层模型输出的顺序合并，
                 若为dataframe，则其index同X，columns形如['s21=1','s21=2','s22=1','s22=2']，其中's21'、
                 's22'为self.stages_的最后一层模型的名称，'1'、'2'表示原始预测概率矩阵的第二、三列（因为概率之和
                 为1，因此未取第一列），若self.type为'regression'，则columns即为self.stages_的最后一层模型的名称。
        '''
        if isinstance(X,pd.DataFrame):
            isdf=True
            index=X.index
            X=X.values
        else:
            isdf=False
        n=X.shape[0]
        for i,stage in enumerate(self.stages_):
            if self.verbose:
                print('stage=%d 转换开始[---------->]'%i)
            stage_X=[]
            for name,clfs in stage:
                if not isinstance(clfs,list):
                    if self.type=='classification':
                        output_tmp=clfs.predict_proba(X,train=train)[:,1:].reshape((n,-1))
                    elif self.type=='regression':
                        output_tmp=clfs.predict(X,train=train).reshape((n,-1))
                    stage_X.append(output_tmp.copy())
                    del output_tmp
                    gc.collect()
                else:
                    if not train:
                        output_tmp=0
                        for clf in clfs:
                            if self.type=='classification':
                                output_tmp+=clf.predict_proba(X)[:,1:].reshape((n,-1))
                            elif self.type=='regression':
                                output_tmp+=clf.predict(X).reshape((n,-1))
                        output_tmp/=float(len(clfs))
                        stage_X.append(output_tmp.copy())
                        del output_tmp
                        gc.collect()
                    else:
                        output = None
                        ks = KFold(self.n_folds, shuffle=False)
                        for j, (train_index, test_index) in enumerate(ks.split(X)):
                            if self.type == 'classification':
                                output_tmp = clfs[j].predict_proba(X[test_index])[:, 1:].reshape((test_index.shape[0], -1))
                            elif self.type == 'regression':
                                output_tmp = clfs[j].predict(X[test_index]).reshape((test_index.shape[0], -1))
                            if j == 0:
                                output = np.full((n, output_tmp.shape[1]), np.nan)
                            output[test_index, :] = output_tmp.copy()
                            del output_tmp
                            gc.collect()
                        stage_X.append(output.copy())
                        del output
                        gc.collect()
                if self.verbose:
                    print('stage=%d model=%s 转换完成'%(i,name))
            X=np.concatenate(stage_X,axis=1)
            del stage_X
            gc.collect()
            if self.verbose:
                print('stage=%d 转换完成[<----------]\n'%i)
        if self.return_array:
            return X
        elif not isdf:
            return X
        else:
            if self.type=='regression':
                columns=[name for name,_ in self.stages_[-1]]
            elif self.type=='classification':
                columns=[]
                for name,_ in self.stages_[-1]:
                    columns.extend(['%s=%d'%(name,k) for k in range(1,self.n_classes_)])
            X=pd.DataFrame(X)
            X.index=index
            X.columns=columns
            return X


class StackingClassifier(BaseEstimator):
    '''
    stacking分类器，使用多个stage转换数据（即StackingTransformer的功能），然后使用combiner作为最终的分类器。

    参数：
    stages: 列表，形如[[('s11',clf11),('s12',clf12)],[('s21',clf21),('s22',clf22),('s23',clf23),('s24',clf24)]]，
            其中每个元素也是列表（注意：元素个数需大于1），对应每个stage，如[('s11',clf11),('s12',clf12)]，其中模型任意
            （与sklearn兼容，有fit、predict_proba即可）。
            同StackingTransformer的stages参数。
    combiner: 最后一层的分类器，与sklearn兼容，有fit、predict_proba或predict即可。
    n_folds: int，交叉验证的折数，只针对StackingTransformer，除非最后一层的combiner属于stacking系列。
    return_array: True则统一返回numpy.ndarray，Fasle则返回形式同X。
                  注意：该参数只针对StackingTransformer，combiner的输出仍然为numpy.ndarray；
                       若combiner属于combiner系列，则该参数也会共享给combiner。
    verbose: 是否打印日志信息。
    kwds: 字典，形式同pipeline的参数，如{'s11__max_depth':4,'s24__C':100}。
          注意：不同stage之间以及同一stage中的模型名称不能重复。对于combiner的参数，形式必须为{'combiner__C':0.1}。

    属性：
    transformer_: 训练好的StackingTransformer，由stages、n_folds、return_array、verbose、kwds参数构成。
    combiner_: 训练好的combiner，最后一层分类器。
    n_classes_: int，类的个数。
    '''

    def __init__(self,stages,combiner,n_folds=5,return_array=False,verbose=False,**kwds):
        self.stages=stages
        self.combiner=combiner
        self.n_folds=n_folds
        self.return_array=return_array
        self.verbose=verbose
        self.kwds=copy.deepcopy(kwds)

    def fit(self,X,y):
        '''
        训练模型，主要是训练出StackingTransformer和combiner。
        :param X: dataframe或二维数组。
        :param y: series（index需要与X对应）或一维数组。
        :return: self。
        '''
        self.n_classes_=len(set(y))
        transformer=StackingTransformer(stages=self.stages,type='classification',n_folds=self.n_folds,
                                        return_array=self.return_array,verbose=self.verbose,**self.kwds)
        combiner=clone(self.combiner)
        if isinstance(combiner,StackingClassifier):
            params={'n_folds':self.n_folds,'return_array':self.return_array,'verbose':self.verbose}
        else:
            params={}
        for k in self.kwds:
            if k.startswith('combiner__'):
                params[k.replace('combiner__','')]=self.kwds[k]
        combiner.set_params(**params)
        if self.verbose:
            print('StackingTransformer训练和转换开始')
        transformer.fit(X,y)
        X=transformer.transform(X,train=True)
        if self.verbose:
            print('StackingTransformer训练和转换完成\n')
            print('combiner训练开始')
        combiner.fit(X,y)
        if self.verbose:
            print('combiner训练完成\n')
        self.transformer_=transformer
        self.combiner_=combiner
        return self

    def predict_proba(self,X,train=False):
        '''
        使用self.transformer_转换数据，self.combiner_预测概率值。
        :param X: dataframe或二维数组。
        :param train: 布尔值，是否为训练阶段，也即输入的X是否为fit阶段的X，训练阶段使用的X和测试阶段使用的X的转换方式应该有所不同，
                      如果均使用求平均的转换方式，训练阶段使用的X得到的转换概率存在过拟合嫌疑。
        :return: 二维numpy.array，self.combiner_的predict_proba的输出结果，每一列代表每一个类别的概率值。
        '''
        if self.verbose:
            print('StackingTransformer转换开始')
        X=self.transformer_.transform(X,train=train)
        if self.verbose:
            print('StackingTransformer转换完成\n')
        if isinstance(self.combiner_,StackingClassifier):
            result=self.combiner_.predict_proba(X,train=train)
        else:
            result=self.combiner_.predict_proba(X)
        if self.verbose:
            print('combiner预测完成\n')
        return result

    def predict(self,X,train=False):
        '''
        使用self.transformer_转换数据，self.combiner_预测类别。
        :param X: dataframe或二维数组。
        :param train: 布尔值，是否为训练阶段，也即输入的X是否为fit阶段的X，训练阶段使用的X和测试阶段使用的X的转换方式应该有所不同，
                      如果均使用求平均的转换方式，训练阶段使用的X得到的转换概率存在过拟合嫌疑。
        :return: 一维numpy.array，self.combiner_的predict的输出结果，代表类别预测结果。
        '''
        if self.verbose:
            print('StackingTransformer转换开始')
        X=self.transformer_.transform(X,train=train)
        if self.verbose:
            print('StackingTransformer转换完成\n')
        if isinstance(self.combiner_,StackingClassifier):
            result=self.combiner_.predict(X,train=train)
        else:
            result=self.combiner_.predict(X)
        if self.verbose:
            print('combiner预测完成\n')
        return result


class StackingRegressor(BaseEstimator):
    '''
    stacking回归，使用多个stage转换数据（即StackingTransformer的功能），然后使用combiner作为最终的回归模型。

    参数：
    stages: 列表，形如[[('s11',clf11),('s12',clf12)],[('s21',clf21),('s22',clf22),('s23',clf23),('s24',clf24)]]，
            其中每个元素也是列表（注意：元素个数需大于1），对应每个stage，如[('s11',clf11),('s12',clf12)]，其中模型任意
            （与sklearn兼容，有fit、predict即可）。
            同StackingTransformer的stages参数。
    combiner: 最后一层的回归模型，与sklearn兼容，有fit、predict即可。
    n_folds: int，交叉验证的折数，只针对StackingTransformer，除非最后一层的combiner属于stacking系列。
    return_array: True则统一返回numpy.ndarray，Fasle则返回形式同X。
                  注意：该参数只针对StackingTransformer，combiner的输出仍然为numpy.ndarray；
                       若combiner属于combiner系列，则该参数也会共享给combiner。
    verbose: 是否打印日志信息。
    kwds: 字典，形式同pipeline的参数，如{'s11__max_depth':4,'s24__C':100}。
          注意：不同stage之间以及同一stage中的模型名称不能重复。对于combiner的参数，形式必须为{'combiner__C':0.1}。

    属性：
    transformer_: 训练好的StackingTransformer，由stages、n_folds、return_array、verbose、kwds参数构成。
    combiner_: 训练好的combiner，最后一层回归模型。
    '''

    def __init__(self,stages,combiner,n_folds=5,return_array=False,verbose=False,**kwds):
        self.stages=stages
        self.combiner=combiner
        self.n_folds=n_folds
        self.return_array=return_array
        self.verbose=verbose
        self.kwds=copy.deepcopy(kwds)

    def fit(self,X,y):
        '''
        训练模型，主要是训练出StackingTransformer和combiner。
        :param X: dataframe或二维数组。
        :param y: series（index需要与X对应）或一维数组。
        :return: self。
        '''
        transformer=StackingTransformer(stages=self.stages,type='regression',n_folds=self.n_folds,
                                        return_array=self.return_array,verbose=self.verbose,**self.kwds)
        combiner=clone(self.combiner)
        if isinstance(combiner,StackingRegressor):
            params={'n_folds':self.n_folds,'return_array':self.return_array,'verbose':self.verbose}
        else:
            params={}
        for k in self.kwds:
            if k.startswith('combiner__'):
                params[k.replace('combiner__','')]=self.kwds[k]
        combiner.set_params(**params)
        if self.verbose:
            print('StackingTransformer训练和转换开始')
        transformer.fit(X,y)
        X=transformer.transform(X,train=True)
        if self.verbose:
            print('StackingTransformer训练和转换完成\n')
            print('combiner训练开始')
        combiner.fit(X,y)
        if self.verbose:
            print('combiner训练完成\n')
        self.transformer_=transformer
        self.combiner_=combiner
        return self

    def predict(self,X,train=False):
        '''
        使用self.transformer_转换数据，self.combiner_预测。
        :param X: dataframe或二维数组。
        :param train: 布尔值，是否为训练阶段，也即输入的X是否为fit阶段的X，训练阶段使用的X和测试阶段使用的X的转换方式应该有所不同，
                      如果均使用求平均的转换方式，训练阶段使用的X得到的转换值存在过拟合嫌疑。
        :return: 一维numpy.array，self.combiner_的predict的输出结果，代表回归结果。
        '''
        if self.verbose:
            print('StackingTransformer转换开始')
        X=self.transformer_.transform(X,train=train)
        if self.verbose:
            print('StackingTransformer转换完成\n')
        if isinstance(self.combiner_,StackingRegressor):
            result=self.combiner_.predict(X,train=train)
        else:
            result=self.combiner_.predict(X)
        if self.verbose:
            print('combiner预测完成\n')
        return result



def test():
    np.random.seed(13)
    X=pd.DataFrame(np.random.randn(1000,10))
    X.columns=['x%d'%i for i in range(X.shape[1])]
    y=pd.Series(np.random.choice([0,1],X.shape[0]))

    from sklearn.linear_model import LogisticRegression,LinearRegression
    from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingClassifier,GradientBoostingRegressor
    from sklearn.svm import SVC,SVR

    stage1=[('rf',RandomForestClassifier()),('gbdt',GradientBoostingClassifier())]
    stage2=[('svm',SVC(probability=True)),('lr1',LogisticRegression())]
    combiner1=LogisticRegression()

    st1=StackingTransformer(stages=[stage1,stage2],type='classification',n_folds=5,return_array=False,verbose=True)
    st1.fit(X,y)
    y1=st1.transform(X,train=False)
    y2=st1.transform(X,train=True)

    sc1=StackingClassifier(stages=[stage1,stage2],combiner=combiner1,n_folds=5,return_array=True,verbose=False)
    sc1.fit(X,y)
    y1=sc1.predict_proba(X,train=False)
    y2=sc1.predict_proba(X,train=True)

    cross_val_score(sc1,X,y,scoring='roc_auc',verbose=1,cv=5)
    cross_val_score(GradientBoostingClassifier(), X, y, scoring='roc_auc', verbose=1, cv=5)

    stage3=[('lr2',LogisticRegression()),('sc1',sc1)]
    combiner2=sc1
    sc2 = StackingClassifier(stages=[stage1, stage2, stage3], combiner=combiner2, n_folds=5, return_array=False, verbose=True)
    sc2.fit(X, y)
    y1 = sc2.predict_proba(X, train=False)
    y2 = sc2.predict_proba(X, train=True)


    stage1 = [('rf', RandomForestRegressor()), ('gbdt', GradientBoostingRegressor())]
    stage2 = [('svm', SVR()), ('lr', LinearRegression())]
    combiner1=LinearRegression()

    clf = StackingTransformer(stages=[stage1, stage2], type='regression', n_folds=5, return_array=False,
                              verbose=True)
    clf.fit(X, y)
    y1 = clf.transform(X, train=False)
    y2 = clf.transform(X, train=True)

    sc1 = StackingRegressor(stages=[stage1, stage2], combiner=combiner1, n_folds=5, return_array=False, verbose=True)
    sc1.fit(X, y)
    y1 = sc1.predict(X, train=False)
    y2 = sc1.predict(X, train=True)
















