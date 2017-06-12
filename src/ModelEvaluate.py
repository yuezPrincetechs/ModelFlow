# -*- coding: utf-8 -*-
"""
模型评估模块。
主要包括：
    计算KS值及对应分割点
    计算KS值，输出对应分割点和累计分布函数曲线图
    计算AUC值，并输出ROC曲线
    画好坏人分数分布对比直方图
    画整体分数的直方图（左Y轴）和每个区间内正类人群占比曲线趋势图（右Y轴）
    根据真实标签和预测概率计算模型指标
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import warnings
warnings.filterwarnings('ignore')

def cal_ks(y,y_prob,pos_label=1,return_split=False,decimals=0):
    '''
    计算KS值及对应分割点。
    y: 一维数组或series，代表真实的标签（{0,1}或{-1,1}）。
    y_prob: 二维数组或dataframe，代表模型预测概率值，默认第二列为预测正类的概率值（即计算所需数值）。
            也可以为一维数组或series，即上述二维数组或dataframe的第二列。
    pos_label: int或字符串，positive所属类别。
    return_split: 是否返回对应分割点。
    decimals: 分割点小数点位数。
    返回KS值或对应分割点（为了能够对接sklearn的评分函数）。
    '''
    y=pd.Series(pd.Series(y).values)
    if len(y_prob.shape)==1:
        y_pred=pd.Series(pd.Series(y_prob).values)
    else:
        y_pred=pd.Series(pd.DataFrame(y_prob).iloc[:,1].values)
    Bad=y_pred[y==pos_label]
    Good=y_pred[y!=pos_label]
    ks, pvalue = stats.ks_2samp(Bad.values, Good.values)
    if not return_split:
        return ks
    crossfreq=pd.crosstab(y_pred.round(decimals),y)
    crossdens = crossfreq.cumsum(axis=0) / crossfreq.sum()
    crossdens['gap'] = abs(crossdens[0] - crossdens[1])
    score_split = crossdens[crossdens['gap'] == crossdens['gap'].max()].index[0]
    return score_split


def plot_ks_cdf(y_true,y_score,pos_label=1,label_map=None,color_map=None,decimals=0,
                xlabel='Score',ylabel='CumSum',fontsize=12,figsize=(18,8),close=True):
    '''
    功能: 计算KS值，输出对应分割点和累计分布函数曲线图
    输入值: 
    y_true: 一维数组或series，代表真实的标签（{0,1}或{-1,1}）。
    y_score: 一维数组或series，代表模型得分（一般为预测正类的概率）。
    pos_label: int或字符串，positive所属类别。
    label_map: 字典，原始类别标签含义，默认为{0:'Good',1:'Bad'}。
    color_map: 字典，原始类别标签对应的颜色，如{0:'g',1:'r'}。
    decimals: 分割点小数点位数。
    xlabel: 字符串，图片xlabel。
    ylabel: 字符串，图片ylabel。
    fontsize: int，字体大小。
    close: 是否关闭图片。
    输出值: 
    字典，键值关系为{'ks': KS值，'split': KS值对应节点，'fig': 累计分布函数曲线图}。
    '''
    if label_map is None:
        label_map={0:'Good',1:'Bad'}
    ks_dict = {}
    y_true=pd.Series(y_true)
    y_score=pd.Series(y_score)
    y_score_dataframe=pd.concat([y_true,y_score],axis=1)
    ks=cal_ks(y_true,y_score_dataframe,pos_label=pos_label,return_split=False,decimals=decimals)
    score_split=cal_ks(y_true,y_score_dataframe,pos_label=pos_label,return_split=True,decimals=decimals)
    
    crossfreq = pd.crosstab(y_score.round(decimals),y_true)
    crossdens = crossfreq.cumsum(axis=0) / crossfreq.sum()
    color=crossdens.columns.map(lambda xx: color_map.get(xx,None))
    crossdens=crossdens.rename(columns=label_map)
    crossdens.columns.name=''
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    crossdens.plot(kind='line',ax=ax,fontsize=fontsize,color=color)
    ax.set_xlabel(xlabel,fontsize=fontsize)
    ax.set_ylabel(ylabel,fontsize=fontsize)
    ax.set_title('CDF Curve (KS=%.2f, SPLIT=%.*f)'%(ks,decimals,score_split),fontsize=fontsize)
    if close:
        plt.close('all')    
    ks_dict['ks'] = ks
    ks_dict['split'] = score_split
    ks_dict['fig'] = fig
    return ks_dict

def plot_roc_auc(y_true,y_score,pos_label=1,color_map=None,figsize=(18,8),fontsize=12,title='ROC Curve',close=True):
    '''
    功能: 计算AUC值，并输出ROC曲线。
    y_true: 一维数组或series，代表真实的标签（{0,1}或{-1,1}）。
    y_score: 一维数组或series或dataframe，代表模型得分（设定为预测为positive的概率或分数）；
             若为dataframe，则每一列代表一个模型得分，列名最好是模型名称，目前如果列数超过6，颜色会重复。
    pos_label: int或字符串，positive所属类别。
    color_map: 字典，键为模型名称，值为roc曲线颜色；若只有一个模型（y_score为一维），则可为代表颜色的字符串。
    figsize: 二元tuple，图片大小。
    fontsize: int，字体大小。
    title: 字符串，图片title。
    close: 是否关闭图片。
    返回ROC曲线图。
    '''
    y_true=(y_true==pos_label).astype(int)
    if len(y_score.shape)==1:
        if color_map is None:
            color='b'
        else:
            color=color_map
        fig,ax=plt.subplots(1,1,figsize=figsize)
        fpr, tpr, threshold = roc_curve(list(y_true),list(y_score))
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, label='AUC = %0.3f' % roc_auc)
        # ax.plot([0, 1], [0, 1], color='r', linestyle='--', alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate',fontsize=fontsize)
        ax.set_ylabel('True Positive Rate',fontsize=fontsize)
        ax.set_title(title,fontsize=fontsize)
        ax.legend(loc="lower right",fontsize=fontsize)
        if close:
            plt.close('all')
    else:
        if color_map is None:
            color_map={}
        colors = cycle(['blue', 'darkorange', 'cyan', 'black', 'indigo', 'seagreen'])
        if not isinstance(y_score,pd.DataFrame):
            raise Exception('y_score若为二维数组，则必须是pandas.DataFrame类型！')
        fig,ax=plt.subplots(1,1,figsize=figsize)
        for col in y_score.columns.tolist():
            fpr, tpr, threshold = roc_curve(list(y_true),list(y_score[col]))
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label='%s, AUC = %0.3f' % (col,roc_auc),
                    color=color_map.get(col,colors.__next__()), alpha=0.7)
        # ax.plot([0, 1], [0, 1], color='r', linestyle='--', alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate',fontsize=fontsize)
        ax.set_ylabel('True Positive Rate',fontsize=fontsize)
        ax.set_title(title,fontsize=fontsize)
        ax.legend(loc="lower right",fontsize=fontsize)
    return fig 

def plot_score_goodbad(y_true,y_score,label_map=None,color_map=None,figsize=(18,8),title=None,
                       xlabel='Score',ylabel='Frequency',fontsize=12,close=True):
    '''
    功能: 画好坏人分数分布对比直方图。
    y_true: 一维数组或series，代表真实的标签（{0,1}或{-1,1}）。
    y_score: 一维数组或series，代表模型得分。
    label_map: 字典，原始类别标签含义，默认为{0:'Good',1:'Bad'}。
    color_map: 字典，原始类别标签对应的颜色，如{0:'g',1:'r'}。
    figsize: tuple，图片大小。
    title: 字符串，图片标题，默认为'Histogram of Score in Good vs. Bad'。
    xlabel: 字符串，图片xlabel。
    ylabel: 字符串，图片ylabel。
    close: 是否关闭图片。
    返回图片对象。
    '''
    if label_map is None:
        label_map={0:'Good',1:'Bad'}
    if color_map is None:
        color_map={}
    result={}
    for key in label_map:
        result[key]=y_score[y_true==key]
    keys=list(result.keys())
    keys.sort()
    colors = cycle(['blue', 'red', 'darkorange', 'cyan', 'black', 'indigo', 'seagreen'])
    fig=plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    for i in keys:
        ax.hist(result[i], bins=100, alpha=0.6,
                color=color_map.get(i,colors.__next__()),label=label_map[i])
    ax.set_xlabel(xlabel,fontsize=fontsize)
    ax.set_ylabel(ylabel,fontsize=fontsize)
    ax.legend(loc='best',fontsize=fontsize)
    if title is None:
        ax.set_title('Histogram of Score in Good vs. Bad',fontsize=fontsize)
    else:
        ax.set_title(title,fontsize=fontsize)
    if close:
        plt.close('all')
    return fig

def plot_score_badratio(y_true,y_score,pos_label=1,bins=10,figsize=(18,8),title=None,xlabel='Score',
                        ylabel='Frequency',ylabel_secondary='Ratio of Positive',fontsize=12,close=True):
    '''
    功能: 画整体分数的直方图（左Y轴）和每个区间内正类人群占比曲线趋势图（右Y轴）。
    y_true: 一维数组或series，代表真实的标签（{0,1}或{-1,1}）。
    y_score: 一维数组或series，代表模型得分。
    pos_label: int或字符串，positive所属类别。
    bins: int（表示分割区间数量）或list（表示分割区间端点取值）。
    figsize: tuple，图片大小。
    title: 字符串，图片标题，默认为'Histogram and Ratio of Score'。
    xlabel: 字符串，图片xlabel。
    ylabel: 字符串，图片ylabel（左轴）。
    ylabel_secondary: 字符串，图片ylabel（右轴）。
    fontsize: int，字体大小。
    close: 是否关闭图片。
    返回图片对象。
    '''
    y_true=(y_true==pos_label).astype(int)
    fig=plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    ax2=ax.twinx()
    _,newbins,_=ax.hist(y_score,bins=bins,alpha=0.6,color='b',label='left')
    x_mid=[]
    ratio=[]
    for i in range(newbins.shape[0]-1):
        x_mid.append((newbins[i+1]+newbins[i])/2.0)
        if i>0:
            ratio.append(y_true[(y_score>newbins[i])&(y_score<=newbins[i+1])].mean())
        else:
            ratio.append(y_true[(y_score >= newbins[i]) & (y_score <= newbins[i + 1])].mean())
    ax2.plot(x_mid,ratio,'-ro',label='right')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='best', fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax2.set_ylabel(ylabel_secondary, fontsize=fontsize)
    if title is None:
        ax.set_title('Histogram and Ratio of Score',fontsize=fontsize)
    else:
        ax.set_title(title,fontsize=fontsize)
    if close:
        plt.close('all')
    return fig

def model_summary(y_true,y_prob,metrics_include=None,metrics_exclude=None,pos_label=1,thresholds=None,quantiles=None):
    '''
    根据真实标签和预测概率计算模型指标，主要包括：
    num_pos、num_neg、TP、FP、TN、FN、accuracy、auc、precision_pos、recall_pos、f1_pos、precision_neg、recall_neg、f1_neg、TPR、FPR。
    :param y_true: 一维数组或series，代表真实的标签。
    :param y_prob: 一维数组或series，代表预测为pos_label的概率。
    :param metrics_include: 字符串列表，需要统计的模型指标，默认全部。
    :param metrics_exclude: 字符串列表，不需要统计的模型指标，默认为空。
    :param pos_label: 数值型或string均可，代表正类标签。
    :param thresholds: None或数值列表。
                       None表示使用分位点列表quantiles获取阈值列表；
                       数值列表表示对列表中的每一个阈值均会计算所有模型指标。
    :param quantiles: 数值列表，0到100之间，代表需要使用的分位点，默认None则使用[0.1*i for i in range(1,1000)]。
    :return: dataframe，index为阈值，columns为各项模型指标。
    '''
    if isinstance(y_true,pd.Series):
        y_true=y_true.values
    if isinstance(y_prob,pd.Series):
        y_prob=y_prob.values
    metrics_all=['num_pos','num_neg','TP','FP','TN','FN','accuracy','auc','precision_pos','recall_pos','f1_pos','precision_neg','recall_neg','f1_neg','TPR','FPR']
    if metrics_exclude is None:
        metrics_exclude=[]
    if metrics_include is None:
        metrics_include=[col for col in metrics_all if col not in metrics_exclude]
    else:
        metrics_include=[col for col in metrics_all if (col not in metrics_exclude) and (col in metrics_include)]

    num_pos=(y_true==pos_label).sum()
    num_neg=y_true.shape[0]-num_pos
    if quantiles is None:
        quantiles=[0.1*i for i in range(1,1000)]
    if thresholds is None:
        thresholds=np.percentile(y_prob,quantiles)
    thresholds=np.sort(thresholds).tolist()
    result = {}
    result['TP'] = np.array([np.sum((y_true == pos_label) & (y_prob > threshold)) for threshold in thresholds]).astype(float)
    result['FP'] = np.array([np.sum((y_true != pos_label) & (y_prob > threshold)) for threshold in thresholds]).astype(float)
    result['TN'] = np.array([np.sum((y_true != pos_label) & (y_prob <= threshold)) for threshold in thresholds]).astype(float)
    result['FN'] = np.array([np.sum((y_true == pos_label) & (y_prob <= threshold)) for threshold in thresholds]).astype(float)
    result['num_pos']=num_pos
    result['num_neg']=num_neg
    result['accuracy']=(result['TP']+result['TN'])/float(y_true.shape[0])
    fpr,tpr,_=roc_curve(y_true,y_prob,pos_label=pos_label)
    result['auc']=auc(fpr,tpr)
    result['precision_pos']=result['TP']/(result['TP']+result['FP'])
    result['recall_pos']=result['TP']/(result['TP']+result['FN'])
    result['f1_pos']=2.0*result['precision_pos']*result['recall_pos']/(result['precision_pos']+result['recall_pos'])
    result['precision_neg']=result['TN']/(result['TN']+result['FN'])
    result['recall_neg']=result['TN']/(result['TN']+result['FP'])
    result['f1_neg']=2.0*result['precision_neg']*result['recall_neg']/(result['precision_neg']+result['recall_neg'])
    result['TPR']=result['recall_pos']
    result['FPR']=1-result['recall_neg']
    result=pd.DataFrame(result)
    result=result.reindex(columns=metrics_include)
    result.index=thresholds
    result.index.name='threshold'
    return result


#==============================================================================
# 测试
#==============================================================================

def test():
    y_true=np.random.choice([0,1],1000)
    y_score=np.zeros(1000)
    y_score[y_true==0]=np.random.randn((y_true==0).sum())*2+5
    y_score[y_true==1]=np.random.randn((y_true==1).sum())*1+1

    # 整体分数分布直方图与正类人群占比趋势图
    fig1=plot_score_badratio(y_true, y_score, pos_label=1, bins=10, figsize=(18, 8), title=None, xlabel='Score',
                             ylabel='Frequency', ylabel_secondary='Ratio of Positive', fontsize=15, close=False)

    # 好坏人分数分布直方图对比
    y_true2 = np.random.choice([0, 1, 2], 1000)
    y_score2 = np.zeros(1000)
    y_score2[y_true2 == 0] = np.random.randn((y_true2 == 0).sum()) * 2 + 5
    y_score2[y_true2 == 1] = np.random.randn((y_true2 == 1).sum()) * 1 + 1
    y_score2[y_true2 == 2] = np.random.randn((y_true2 == 2).sum()) * 0.5 + 0.5
    fig2=plot_score_goodbad(y_true2, y_score2, label_map={0:'positive',1:'negative',2:'other'},
                            color_map={0:'green',1:'red',2:'yellow'},figsize=(18, 8), title='Three Types',
                            xlabel='Score', ylabel='Frequency', fontsize=12, close=False)

    # 模型评估summary
    result=model_summary(y_true,-y_score,pos_label=1,thresholds=[0],quantiles=None)





