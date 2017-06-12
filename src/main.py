# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from importlib import reload
from sklearn.preprocessing import Imputer, StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import cross_validation,linear_model,grid_search
from sklearn import metrics
import os

#导入相应模块
import src.BasicDataStruct as BasicDataStruct
import src.DataAnalysis as DataAnalysis
import src.discretize as discretize
import src.continuous as continuous
import src.FeatureSelection as FeatureSelection
import src.ModelEvaluate as ModelEvaluate
import src.utils.utils as utils

path='/Users/yuez/Desktop/工作/普林科技项目代码/ModelFlow/resource/'

#%%读入数据并区分出X和Y
data=pd.read_csv(path+'credit_card_data.csv',encoding='utf8',index_col=[0])
Y=data['SeriousDlqin2yrs'].copy()
X=data[[col for col in data.columns.tolist() if col!='SeriousDlqin2yrs']].copy()

#需要区分三类变量：连续型、0-1离散型、多取值离散型
feature_cate_two=[]
feature_cate_mult=[]

#%%创建基础数据对象
model=BasicDataStruct.BasicDataStruct(X=X,Y=Y,
                                      feature_cate_two=feature_cate_two,
                                      feature_cate_mult=feature_cate_mult,
                                      model_type='classification')
print('原始数据：')
print(model.current_state())

#%%描述性统计、变量分析

#描述性统计量、缺失值、异常值、相关性
outfile=path+'基本统计/'
utils.ensure_directory(outfile)

stat_describe=DataAnalysis.stat_df(model.X)
stat_error=DataAnalysis.error_df(model.X)
stat_corr,fig_corr=DataAnalysis.cor_df(model.X,xticklabels=False,yticklabels=False,close=False)

stat_describe.to_excel(outfile+'stat_describe.xlsx')
stat_error.to_excel(outfile+'stat_error.xlsx')
stat_corr.to_excel(outfile+'stat_corr.xlsx')
fig_corr.savefig(outfile+'fig_corr.png')

#单变量分布
outfile=path+'单变量分布/'
utils.ensure_directory(outfile)

result=DataAnalysis.dist_plot(model.X,
                              feature_cate=model.feature_cate_two+model.feature_cate_mult,
                              close=True,show_last=True,verbose=True)
result_y=DataAnalysis.dist_plot(model.Y.to_frame(),
                                feature_cate=[model.Y.name],
                                close=True,show_last=True,verbose=True)
result.update(result_y)

for col in result:
    if len(result[col])==2:
        result[col][0].savefig(outfile+'%s_直方密度图.png'%col)
        result[col][1].savefig(outfile+'%s_箱线图.png'%col)
    else:
        result[col][0].savefig(outfile+'%s_柱形图.png'%col)

#单变量X与Y的分析图
outfile=path+'单变量X与Y的分析图/'

result=DataAnalysis.plot_singleXY_Mean(model.X,model.Y,
                                       feature_cate=model.feature_cate_two+model.feature_cate_mult,
                                       normalize=True,close=True,show_last=True,verbose=True)

for col in result:
    result[col].savefig(outfile+'%s.png'%col)

#两变量X与Y的分析图
outfile=path+'两变量X与Y的分析图/'

result=DataAnalysis.plot_doubleXY_Mean(model.X,Y_cate=model.Y,
                                       feature_cate=model.feature_cate_two+model.feature_cate_mult,
                                       backend='seaborn',close=True,show_last=True,verbose=True)

for cols in result:
    result[cols].savefig(outfile+'%s-%s.png'%(cols[0],cols[1]))


#%%提取离散化特征（缺失值会被当作一类）
clf_dis=discretize.QuantileDiscretizer(quantiles=[10*i for i in range(1,10)],fill_na='Missing',return_numeric=False)
Xnew_dis=clf_dis.fit_transform(model.X[model.feature_cont])

model_dis=BasicDataStruct.BasicDataStruct(X=Xnew_dis,Y=None,
                                          feature_cate_two=[],
                                          feature_cate_mult=Xnew_dis.columns.tolist(),
                                          model_type='classification')
print('连续特征离散化：')
print(model_dis.current_state())

#%%原始变量缺失填补
clf_imputer_cont=Imputer(strategy='mean')
clf_imputer_cate=Imputer(strategy='most_frequent')

if model.feature_cont!=[]:
    model.X[model.feature_cont]=clf_imputer_cont.fit_transform(model.X[model.feature_cont])
if model.feature_cate_two+model.feature_cate_mult!=[]:
    model.X[model.feature_cate_two+model.feature_cate_mult]=clf_imputer_cate.fit_transform(model.X[model.feature_cate_two+model.feature_cate_mult])

print('原始数据缺失填补：')
print(model.current_state())

#%%原始离散变量独热编码
clf_onehot=continuous.OneHotContinuous(drop_others=True)

if model.feature_cate_mult!=[]:
    Xnew_onehot=clf_onehot.fit_transform(model.X[model.feature_cate_mult])
    model.delete(model.feature_cate_mult)
    model.add(Xnew_onehot,feature_cate_two=Xnew_onehot.columns.tolist())

print('原始离散变量独热编码：')
print(model.current_state())

#%%原始变量利用决策树生成交叉项
from feature_eng.tree_based.dtree_discretizer import ClassificationDTDiscretizer

clf_cross=ClassificationDTDiscretizer(columns=model.X.columns.tolist(),inters_only=True)
Xnew_cross=clf_cross.fit_transform(model.X,model.Y)
Xnew_cross=pd.DataFrame(Xnew_cross,columns=clf_cross.get_column_names(),index=model.X.index)

model_cross=BasicDataStruct.BasicDataStruct(X=Xnew_cross,Y=None,
                                            feature_cate_two=Xnew_cross.columns.tolist(),
                                            feature_cate_mult=[],
                                            model_type='classification')

print('生成交叉项：')
print(model_cross.current_state())

#%%原始连续变量变换
clf_log=FunctionTransformer(np.log1p)

model.X[model.feature_cont]=clf_log.fit_transform(model.X[model.feature_cont])

print('原始连续变量log变换：')
print(model.current_state())



#%%合并数据
model.addObj(model_dis,replace=False,ignore=False,suffix='_dis')
model.addObj(model_cross,replace=False,ignore=True)

print('变换后的原始变量、连续变量离散化结果、交叉项合并：')
print(model.current_state())


#%%计算离散型变量的IV值
clf_woe=continuous.WoeContinuous(feature_names=model.feature_cate_two+model.feature_cate_mult)
clf_woe.fit(model.X,model.Y)

#计算IV值并画出前5的直方图
IV,fig_IV=clf_woe.plot_iv(top=5,rot=90,close=True)

#可视化每个变量每个类别的WOE
figs_woe=clf_woe.plot(rot=30,close=True,show_last=False)

fig_IV.show()
figs_woe[list(figs_woe.keys())[-1]].show()

outfile=path+'离散变量WOE分布/'
utils.ensure_directory(outfile)

for feature in figs_woe:
    figs_woe[feature].savefig(outfile+'%s.png'%feature.replace('<=',' LE ').replace('>',' GT ').replace('<',' LT ').replace('>=',' GE '))


#%%对多取值离散变量进行独热编码
clf_onehot_dis=continuous.OneHotContinuous(drop_others=True)

if model.feature_cate_mult!=[]:
    Xnew_onehot=clf_onehot_dis.fit_transform(model.X[model.feature_cate_mult])
    model.delete(model.feature_cate_mult)
    model.add(Xnew_onehot,feature_cate_two=Xnew_onehot.columns.tolist())

print('独热编码：')
print(model.current_state())


#%%特征筛选
clf_feature_selection=FeatureSelection.SklearnSelector(SelectFromModel(RandomForestClassifier(),threshold='median'))
clf_feature_selection.fit(model.X,model.Y)

model.select(clf_feature_selection.feature_selected)

print('特征筛选：')
print(model.current_state())

#%%构建模型，调参
params={'penalty':('l2',),
        'C':(0.1,1,10),
        'class_weight':({0:1,1:5},'balanced')}

lr=linear_model.LogisticRegression(random_state=13)
clf_grid_search=grid_search.GridSearchCV(lr,params,verbose=0,scoring='roc_auc')
clf_grid_search.fit(model.X,model.Y)

best_params=clf_grid_search.best_params_

print('格点搜索得到最优参数：',best_params)

#%%交叉验证获取平均得分
lr=linear_model.LogisticRegression(random_state=13)
best_params={'C': 0.1, 'penalty': 'l2', 'class_weight': 'balanced'}
lr.set_params(**best_params)
ks_scoring=metrics.make_scorer(ModelEvaluate.cal_ks,needs_proba=True,return_split=False,decimals=4)

print('五折交叉验证AUC值：',cross_validation.cross_val_score(lr,model.X,model.Y,scoring='roc_auc',cv=5))
print('五折交叉验证KS值：',cross_validation.cross_val_score(lr,model.X,model.Y,scoring=ks_scoring,cv=5))

#%%最终模型效果评估
lr.fit(model.X,model.Y)

score=lr.predict_proba(model.X)[:,0]
score=pd.Series(score)
score.index=model.Y.index

y_pred=lr.predict(model.X)
y_pred=pd.Series(y_pred)
y_pred.index=model.Y.index

result_ks=ModelEvaluate.plot_ks_cdf(model.Y,score,decimals=4,close=False)

result_roc=ModelEvaluate.plot_roc_auc(model.Y,score,close=False)

result_hist=ModelEvaluate.plot_hist_score(model.Y,score,close=False)

print(metrics.classification_report(model.Y,y_pred))



