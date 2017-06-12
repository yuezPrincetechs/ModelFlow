# -*- coding: utf-8 -*-
"""
数据分析模块。
主要包括: 
    描述性统计量、缺失值、异常值统计
    相关系数计算、热力图
    单变量分析图: 直方密度图、箱线图（连续型），柱形图（离散型）
    单变量与Y（单一）的分析图: 样本数量和Y组内比例柱形图（离散型Y），均值柱形图（连续型Y）
    单变量与Y（可多个）的分析图: 样本数量和Y均值柱形图（离散型Y或连续型Y）
    两个变量X与Y（可多个）的分析图: 0-1离散型Y创建1的占比热力图；连续型Y创建均值热力图
"""
import pandas as pd
import numpy as np
try:
    import seaborn as sns
except:
    sns=None
import matplotlib
##Linux下需要设置参数使得图片不显示，matplotlib.
import matplotlib.pyplot as plt
import src.discretize as discretize
import src.utils.utils as utils
import copy

def stat_series(data,numeric=None,quantiles=None,num_most=5,index='English'):
    '''
    对series计算所有描述统计。
    data: series。
    numeric: True表示字段类型为'numeric'，Fasle表示字段类型为'categorical'或'binary'，None表示从数据类型中推断。
             需要注意的是，若numeric为True而数据类型为字符型，则字段类型也一定是'categorical'或'binary'。
    quantiles: 列表，需要统计的分位点，0到100之间，默认为[25,50,75]。
    num_most: int，统计出现最多的前num_most个值。
    index: {'English','Chinese'}。

    返回描述统计的series，包含如下部分: 
    中文名: 字段类型、总记录数、非重复记录数、缺失记录数、缺失比例、均值、标准差、最小值、各分位点（如25%分位点,50%分位点,75%分位点）、最大值、
    出现第N多的值、出现第N多的值占比（N从1到num_most）；
    英文名: type,count,count_unique,missing_count,missing_ratio,mean,std,min,各分位点（如quantile25,quantile50,quantile75）,max,
    valuemostN,freqmostN（N从1到num_most）
    
    '''
    if quantiles is None:
        quantiles=[25,50,75]
    index_english=['type','count','count_unique','missing_count','missing_ratio','mean','std','min']
    index_english.extend(['quantile%g'%i for i in quantiles])
    index_english.extend(['max'])
    for i in range(num_most):
        index_english.extend(['valuemost%d'%(i+1),'freqmost%d'%(i+1)])
    index_mapping={'type': '字段类型',
                   'count': '总记录数',
                   'count_unique': '非重复记录数',
                   'missing_count': '缺失记录数',
                   'missing_ratio': '缺失比例',
                   'mean': '均值',
                   'std': '标准差',
                   'min': '最小值',
                   'max': '最大值'
                   }
    for i in quantiles:
        index_mapping['quantile%g'%i]='%g%%分位点'%i
    for i in range(num_most):
        index_mapping['valuemost%d'%(i+1)]='出现第%d多的值'%(i+1)
        index_mapping['freqmost%d'%(i+1)]='出现第%d多的值占比'%(i+1)
    result={}
    result['count']=data.shape[0]
    result['missing_count']=data.isnull().sum()
    result['missing_ratio']=result['missing_count']/float(result['count'])
    data=data.dropna()
    if numeric==None or numeric==True:
        if data.dtype in [np.dtype('int8'),np.dtype('int16'),np.dtype('int32'),np.dtype('int64'),np.dtype('float16'),np.dtype('float32'),np.dtype('float64')]:
            result['type']='numeric'
            result['mean']=data.mean()
            result['std']=data.std()
            result['min']=data.min()
            for i in quantiles:
                result['quantile%g'%i]=data.quantile(i/100.0)
            result['max']=data.max()
            value_counts=data.value_counts(normalize=True)
            result['count_unique']=value_counts.shape[0]
            if result['count_unique']<=2:
                result['type']='binary'
            for i in range(num_most):
                try:
                    result['valuemost%d'%(i+1,)]=value_counts.index[i]
                    result['freqmost%d'%(i+1,)]=value_counts.iloc[i]
                except:
                    break
        else:
            result['type']='categorical'
            value_counts=data.value_counts(normalize=True)
            result['count_unique'] = value_counts.shape[0]
            if result['count_unique']<=2:
                result['type']='binary'
            for i in range(num_most):
                try:
                    result['valuemost%d'%(i+1,)]=value_counts.index[i]
                    result['freqmost%d'%(i+1,)]=value_counts.iloc[i]
                except:
                    break
    else:
        result['type'] = 'categorical'
        value_counts = data.value_counts(normalize=True)
        result['count_unique'] = value_counts.shape[0]
        if result['count_unique'] <= 2:
            result['type'] = 'binary'
        for i in range(num_most):
            try:
                result['valuemost%d' % (i + 1,)] = value_counts.index[i]
                result['freqmost%d' % (i + 1,)] = value_counts.iloc[i]
            except:
                break
    result=pd.Series(result)
    result=result.reindex(index_english)
    if index=='Chinese':
        result=result.rename(index=index_mapping)
    return result

def stat_df(data,cols=None,cols_cate=None,quantiles=None,num_most=5,index='English'):
    '''
    对每一列计算所有描述统计。
    data: dataframe。
    cols: 选取字段，list类型，默认为data的所有列。
    cols_cate: 列表，表示离散型字段，默认为空（根据数据类型推断）。
    quantiles: 列表，需要统计的分位点，0到100之间，默认为[25,50,75]。
    num_most: int，统计出现最多的前num_most个值。
    index: {'English','Chinese'}。
    返回描述统计的dataframe，index为变量名，columns包含如下部分: 
    中文名: 字段类型、总记录数、非重复记录数、缺失记录数、缺失比例、均值、标准差、最小值、各分位点（如25%分位点,50%分位点,75%分位点）、最大值、
    出现第N多的值、出现第N多的值占比（N从1到num_most）；
    英文名: type,count,count_unique,missing_count,missing_ratio,mean,std,min,各分位点（如quantile25,quantile50,quantile75）,max,
    valuemostN,freqmostN（N从1到num_most）
    
    '''
    if quantiles is None:
        quantiles=[25,50,75]
    result=[]
    if cols is None:
        cols=data.columns.tolist()
    if cols_cate is None:
        cols_cate=[]
    for col in cols:
        numeric=False if col in cols_cate else None
        tmp=stat_series(data[col],numeric=numeric,num_most=num_most,quantiles=quantiles,index=index)
        tmp.name=col
        result.append(tmp.copy())
    result=pd.concat(result,axis=1).T
    if index=='English':
        result.index.name='Variable'
    else:
        result.index.name='变量名'
    return result

def stat_df_categorical(data):
    '''
    针对离散型变量计算描述统计，目前包含：变量名、总记录数、非重复记录数、缺失记录数、缺失比例、取值、样本量及占比。
    :param data: dataframe，均为离散型数据。
    :return: dataframe，列名为：变量名、总记录数、非重复记录数、缺失记录数、缺失比例、取值、样本量、占比。
             由于每个变量有多个取值，因此每个变量会有多行。
    '''
    result=[]
    for col in data.columns.tolist():
        tmp1={}
        tmp1['变量名']=col
        tmp1['总记录数']=data[col].shape[0]
        tmp1['缺失记录数']=data[col].isnull().sum()
        tmp1['缺失比例']=tmp1['缺失记录数']/float(tmp1['总记录数'])
        value_counts=data[col].value_counts(normalize=False)
        value_counts.index.name='取值'
        value_counts.name='样本量'
        value_counts=value_counts.reset_index()
        value_counts['占比']=value_counts['样本量']/float(value_counts['样本量'].sum())
        tmp1['非重复记录数']=value_counts.shape[0]
        tmp1=pd.DataFrame(pd.Series(tmp1)).T
        if value_counts.shape[0]>1:
            tmp2=pd.DataFrame({'变量名':[col]*(value_counts.shape[0]-1),
                               '总记录数':[np.nan]*(value_counts.shape[0]-1),
                               '非重复记录数': [np.nan] * (value_counts.shape[0] - 1),
                               '缺失记录数': [np.nan] * (value_counts.shape[0] - 1),
                               '缺失比例': [np.nan] * (value_counts.shape[0] - 1)})
        else:
            tmp2=pd.DataFrame()
        tmp1=pd.concat([tmp1,tmp2],axis=0)
        tmp1=tmp1.reset_index(drop=True)
        value_counts=value_counts.reset_index(drop=True)
        tmp=pd.concat([tmp1,value_counts],axis=1)
        result.append(tmp.copy())
    result=pd.concat(result,axis=0,ignore_index=True)
    result=result.reindex(columns=['变量名','总记录数','非重复记录数','缺失记录数','缺失比例',
                                   '取值','样本量','占比'])
    return result

#Abnormal value
def error_df(data, cols=None):
    '''
    功能: 创建异常值表格，根据zscore的绝对值是否大于3.5判断
    输入值: 
    data: 原始数据，dataframe类型
    cols: 选取字段，list类型，默认为data的所有列
    输出值: 
    error_data: 异常值数据，dataframe类型
    '''
    if cols is None:
        cols=list(data.columns)
    error_data = pd.DataFrame(index=cols, columns=['ErrorNum','ErrorPercent'])
    for column in cols:
        error_series = data[column]
        zscore = (error_series - error_series.mean()) / error_series.std()
        error_count = (zscore.abs() > 3.5).sum()
        error_percent = error_count.astype(float) / len(error_series)
        error_data.ix[column,:] = error_count, error_percent
    return error_data

#Correlation
def cor_df(data, cols=None, xticklabels=False, yticklabels=False, close=True):
    '''
    功能: 创建相关性矩阵和热点图
    输入值: 
    data: 原始数据，dataframe类型
    cols: 选取字段，list类型，默认为data的所有列
    close: 是否关闭生成的图
    输出值: 
    cormat: 相关性矩阵，dataframe类型
    heatmap: 热点图，fig类型
    '''
    if cols is None:
        cols=list(data.columns)
    corrmat = data[cols].corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.set(context='paper', font='monospace')
    sns.heatmap(corrmat, vmax=0.8, square=True, ax=ax, xticklabels=xticklabels, yticklabels=yticklabels)
    ax.set_title('Heatmap of Correlation Matrix')
    if close:
        plt.close('all')
    return corrmat, fig


#Distribution
def dist_plot(data, cols=None, feature_cate=None, figsize=(18,8), close=True, show_last=True, verbose=False):
    '''
    功能: 连续型变量创建直方密度图、箱线图；离散型变量创建柱形图
    输入值: 
    data: 原始数据，dataframe类型
    cols: 选取字段，list类型，默认为data的所有列
    feature_cate: 离散型变量字段，list类型，默认为空
    close: 是否关闭生成的图
    show_last: 是否展示最后一幅图
    verbose: 是否打印日志
    输出值: 
    fig_dict: 分布图字典；cols为key；fig_list为value，
              连续型fig_list[0]为直方密度图，连续型fig_list[1]为箱线图，
              离散型fig_list[0]为柱形图
    '''
    if cols is None:
        cols=list(data.columns)
    if feature_cate is None:
        feature_cate=[]
    fig_dict = {}
    fig_dict = fig_dict.fromkeys(cols)
    for i,column in enumerate(cols):
        if verbose:
            print(column)
        fig_list = []
        if column not in feature_cate:
            dist = data[column]
            fig1 = plt.figure(figsize=figsize)
            ax = fig1.add_subplot(111)
            dist.plot(kind='hist', bins=20, alpha=0.3, color='b', ax=ax, legend=False)
            ax2 = ax.twinx()
            dist.plot(kind='kde', style='k--', ax=ax2, legend=False, title='Histogram and Density of %s' % column)
            ax.grid()
            ax.set_xlabel('%s' % column)
            ax.set_ylabel('Frequency')
            ax2.set_ylabel('Density')
            fig2 = plt.figure(figsize=figsize)
            ax = fig2.add_subplot(111)
            dist.plot(kind='box', color='b', legend=False, ax=ax)
            ax.set_title('Boxplot of %s' % column)
            if close:
                plt.close('all')
            fig_list.append(fig1)
            fig_list.append(fig2)
            fig_dict[column] = fig_list.copy()
        else:
            dist = data[column]
            fig3 = plt.figure(figsize=figsize)
            ax = fig3.add_subplot(111)
            dist.value_counts().plot(kind='bar', color='b', legend=False, ax=ax)
            ax.set_ylabel('Frequency')
            ax.set_title('Barchart of %s' % column)
            if close:
                plt.close('all')
            fig_list.append(fig3)
            fig_dict[column] = fig_list.copy()
    if show_last:
        try:
            fig1
            fig1.show()
        except:
            pass
        try:
            fig2
            fig2.show()
        except:
            pass
        try:
            fig3
            fig3.show()
        except:
            pass
    return fig_dict

#单变量X与Y的分析图（Y组内比例，单一Y）
def plot_singleXY_PercentInY(X, cols=None, Y_cont=None, Y_cate=None, feature_cate=None, quantiles=None, cuts=None,
                             pattern='\((.*?),',str_nopattern=None,xlabel=None,ylabel=None,color_map=None,legend_map=None,
                             fontsize=12,figsize=(18,8), close=True, show_last=True, verbose=False):
    '''
    功能: 单一X与单一Y的分析图（Y组内比例）。0-1离散型Y创建数量和Y组内比例柱形图；连续型Y创建数量和Y均值柱形图
    输入值: 
    X: 原始数据，dataframe类型
    cols: 选取字段，list类型，默认为data的所有列
    Y_cont: 连续型Y值，Series或一维np.array
    Y_cate: 0-1离散型Y值，Series或一维np.array
    feature_cate: 离散型X变量字段，list类型，默认为空
    quantiles: dict，键为变量名，值为list或一维数组，用于指定连续变量离散化的分位点，默认所有连续变量的分位点为[20*i for i in range(1,5)]
    cuts：dict，键为变量名，值为list或一维数组，用于直接指定连续变量离散化的分割点，优先级高于quantiles
    pattern: 正则表达式，用于匹配横轴标签字符串，使其按照该正则表达式提取后的数值排序
    str_nopattern: 字典，键为变量名（或变量位置），值为列表，表示未匹配pattern字符串的正常顺序
    xlabel: 字符串，表示纵轴标签
    ylabel: 二元列表，表示各个子图的纵轴标签，离散型Y默认为['Count of Samples','Percent in Category of Y']，连续型Y默认为['Count of Samples','Mean of Y']
    color_map: 字典，表示离散型Y原始取值对应的柱形图颜色，如{1:'red',0:'blue'}，只针对离散型Y。
    legend_map: 字典，表示离散型Y原始取值与图例的对应关系，如{1:'bad',0:'good'}，只针对离散型Y。
    fontsize: int，字体大小
    close: 是否关闭生成的图
    show_last: 是否展示最后一幅图
    verbose: 是否打印日志
    输出值: 
    fig_dict: X～Y关系图字典；cols为key；fig（上下两个子图）为value；
              离散型Y上面那幅子图为数量柱形图，下面那幅子图为Y组内比例柱形图；
              连续型Y上面那幅子图为数量柱形图，下面那幅子图为Y均值柱形图
    '''
    data=X.copy()
    if cols is None:
        cols=list(data.columns)
    if feature_cate is None:
        feature_cate=[]
    if quantiles is None:
        quantiles={}
    if cuts is None:
        cuts={}
    if str_nopattern is None:
        str_nopattern={}
    if xlabel is None:
        xlabel=''
    if legend_map is None:
        legend_map={}
    if color_map is None:
        color_map={}
    for key in quantiles:
        quantiles[key]=np.sort(np.unique(quantiles[key])).tolist()
    for key in cuts:
        cuts[key]=np.sort(np.unique(cuts[key]))
    q_default=[20*i for i in range(1,5)]
    fig_dict = {}
    fig_dict = fig_dict.fromkeys(cols)
    if (Y_cont is None) and (Y_cate is None):
        raise Exception('Y值未给定！')
    if (Y_cont is None) and (Y_cate is None):
        raise Exception('连续型和离散型Y值只能给定一种！')
    if Y_cate is not None:
        Y_cate=pd.Series(Y_cate)
        if ylabel is None:
            ylabel=['Count of Samples','Percent in Category of Y']
        for i,column in enumerate(cols):
            if verbose:
                print(column)
            if column not in feature_cate:
                clf = discretize.QuantileDiscretizer(quantiles=quantiles.get(column,q_default),return_numeric=False,fill_na='Missing')
                if column in cuts.keys():
                    clf.cuts=cuts[column]
                else:
                    clf.fit(data[column])
                data[column] = clf.transform(data[column])
            data[column]=data[column].fillna('Missing')
            count=pd.crosstab(data[column],Y_cate)
            count.columns.name=''
            count.index.name=column
            count=count.reindex(utils.sort(count.index.tolist(),ascending=True,pattern=pattern,
                                           str_nopattern=str_nopattern.get(column,None),converter=float))
            color=count.columns.map(lambda xx: color_map.get(xx,None)).tolist()
            count.columns=count.columns.map(lambda xx: legend_map.get(xx,xx))
            ratio=count/count.sum()
            
            fig,axes=plt.subplots(2,1,sharex=True,figsize=figsize)
            count.plot(kind='bar',ax=axes[0],rot=0,fontsize=fontsize,color=color)
            axes[0].set_ylabel(ylabel[0],fontsize=fontsize)
            axes[0].set_title(column,fontsize=fontsize)
            axes[0].legend(loc='best',fontsize=fontsize)
            
            ratio.plot(kind='bar',ax=axes[1],rot=0,color=color)
            axes[1].set_xlabel(xlabel,fontsize=fontsize)
            axes[1].set_ylabel(ylabel[1],fontsize=fontsize)
            axes[1].legend(loc='best',fontsize=fontsize)
                        
            if close:
                plt.close('all')
            fig_dict[column] = fig
    else:
        Y_cont=pd.Series(Y_cont)
        if ylabel is None:
            ylabel=['Count of Samples','Mean of Y']
        for i,column in enumerate(cols):
            if verbose:
                print(column)
            if column not in feature_cate:
                clf = discretize.QuantileDiscretizer(quantiles=quantiles.get(column,q_default),return_numeric=False,fill_na='Missing')
                if column in cuts.keys():
                    clf.cuts=cuts[column]
                else:
                    clf.fit(data[column])
                data[column] = clf.transform(data[column])
            data[column] = data[column].fillna('Missing')
            count=Y_cont.groupby(data[column]).count()
            count.name=''
            count=count.reindex(utils.sort(count.index.tolist(),ascending=True,pattern=pattern,
                                           str_nopattern=str_nopattern.get(column, None),converter=float))
            
            ratio=Y_cont.groupby(data[column]).mean()
            ratio.name=''
            ratio=ratio.reindex(utils.sort(ratio.index.tolist(),ascending=True,pattern='\((.*?),',converter=float))
            
            fig,axes=plt.subplots(2,1,sharex=True,figsize=figsize)
            count.plot(kind='bar',ax=axes[0],rot=0,fontsize=fontsize)
            axes[0].set_ylabel(ylabel[0],fontsize=fontsize)
            axes[0].set_title(column,fontsize=fontsize)
            axes[0].legend(loc='best',fontsize=fontsize)
            
            ratio.plot(kind='bar',ax=axes[1],rot=0,fontsize=fontsize)
            axes[1].set_xlabel(xlabel,fontsize=fontsize)
            axes[1].set_ylabel(ylabel[1],fontsize=fontsize)
            axes[1].legend(loc='best',fontsize=fontsize)
            
            if close:
                plt.close('all')
            fig_dict[column] = fig
    if show_last:
        try:
            fig
            fig.show()
        except:
            pass
    return fig_dict

#单变量X与Y的分析图（均值，多个Y）
def plot_singleXY_Mean(X, Y, cols=None, feature_cate=None, normalize=True, quantiles=None, cuts=None,
                       pattern='\((.*?),',str_nopattern=None,ylabel=None,fontsize=12,
                       figsize=(18,8), close=True, show_last=True, verbose=False):
    '''
    功能: 单一X与多个Y的分析图（Y均值）。0-1离散型Y创建数量和每个X类别中的1占比柱形图；连续型Y创建数量和Y均值柱形图。本质上都是均值柱形图。
    输入值: 
    X: 原始数据，dataframe类型
    Y: 连续型或0-1离散型Y值，Series或一维np.array或DataFrame
    cols: 选取字段，list类型，默认为rawdata的所有列
    feature_cate: 离散型X变量字段，list类型，默认为空
    normalize: 是否对样本数量作归一化（即使用样本占比）
    quantiles: dict，键为变量名，值为list或一维数组，用于指定连续变量离散化的分位点，默认所有连续变量的分位点为[20*i for i in range(1,5)]
    cuts：dict，键为变量名，值为list或一维数组，用于直接指定连续变量离散化的分割点，优先级高于quantiles
    pattern: 正则表达式，用于匹配横轴标签字符串，使其按照该正则表达式提取后的数值排序
    str_nopattern: 字典，键为变量名（或变量位置），值为列表，表示未匹配pattern字符串的正常顺序
    ylabel: 二元列表，表示各个子图的纵轴标签，默认为['Count of Samples','Mean of Y']
    fontsize: int，字体大小
    close: 是否关闭生成的图
    show_last: 是否展示最后一幅图
    verbose: 是否打印日志
    输出值: 
    fig_dict: X～Y关系图字典；cols为key；fig（上下两个子图）为value；
              上面那幅子图为数量柱形图，下面那幅子图为Y均值柱形图。
    '''
    data=X.copy()
    legend=True
    if str_nopattern is None:
        str_nopattern={}
    if cols is None:
        cols=list(data.columns)
    if feature_cate is None:
        feature_cate=[]
    if quantiles is None:
        quantiles={}
    if cuts is None:
        cuts={}
    for key in quantiles:
        quantiles[key]=np.sort(np.unique(quantiles[key])).tolist()
    for key in cuts:
        cuts[key]=np.sort(np.unique(cuts[key]))
    q_default=[20*i for i in range(1,5)]
    fig_dict = {}
    Ynew=pd.DataFrame(Y)
    if isinstance(Y,np.ndarray) or len(Y.shape)==1:
        legend=False
    if ylabel is None:
        ylabel=['Count of Samples','Mean of Y']
    for i,col in enumerate(cols):
        if verbose:
            print(col)
        if col not in feature_cate:
            clf = discretize.QuantileDiscretizer(quantiles=quantiles.get(col, q_default), return_numeric=False,
                                                 fill_na='Missing')
            if col in cuts.keys():
                clf.cuts = cuts[col]
            else:
                clf.fit(data[col])
            data[col] = clf.transform(data[col])
        data[col]=data[col].fillna('Missing')
        value_count=Ynew.groupby(data[col]).count()
        if normalize:
            value_count=value_count/value_count.sum(axis=0)
        value_count=value_count.reindex(utils.sort(value_count.index.tolist(),ascending=True,
                                                   pattern=pattern,str_nopattern=str_nopattern.get(col,None),converter=float))
        value_mean=Ynew.groupby(data[col]).mean()
        value_mean=value_mean.reindex(utils.sort(value_mean.index.tolist(),ascending=True,
                                                 pattern=pattern,str_nopattern=str_nopattern.get(col,None),converter=float))
        
        fig,axes=plt.subplots(2,1,sharex=True,figsize=figsize)
        value_count.plot(kind='bar',rot=30,ax=axes[0],legend=legend,fontsize=fontsize)
        value_mean.plot(kind='bar',rot=30,ax=axes[1],legend=legend,fontsize=fontsize)
        axes[0].set_ylabel(ylabel[0],fontsize=fontsize)
        axes[0].set_title(col,fontsize=fontsize)

        axes[1].set_xlabel('')
        axes[1].set_ylabel(ylabel[1],fontsize=fontsize)
        
        fig_dict[col]=fig
        if close:
            plt.close('all')
    if show_last:
        try:
            fig
            fig.show()
        except:
            pass
    
    return fig_dict

def heatmap(data,ax,xlabel=None,ylabel=None,xticklabels=None,yticklabels=None,title=None,fontsize=12):
    '''
    使用matplotlib.pyplot.pcolor画热力图。
    返回二元组(pc,ax)，其中pc可以作为matplotlib.pyplot.colorbar的第一个参数mappable。
    '''
    pc=ax.pcolor(data,cmap=plt.cm.Blues)
    if xlabel is not None:
        ax.set_xlabel(xlabel,fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel,fontsize=fontsize)
    ax.set_xticks(np.arange(data.shape[1])+0.5,minor=False)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels,minor=False,fontsize=fontsize)
    ax.set_yticks(np.arange(data.shape[0])+0.5,minor=False)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels,minor=False,fontsize=fontsize)
    if title is not None:
        ax.set_title(title,fontsize=fontsize)
    return pc,ax


#两个变量X与Y的分析图
def plot_doubleXY_Mean(X, cols_h=None, cols_v=None, Y_cont=None, Y_cate=None, feature_cate=None, quantiles=None, cuts=None,
                       pattern='\((.*?),',str_nopattern=None,fontsize=12,
                       backend='seaborn', figsize=(18,8), close=True, show_last=True, verbose=False):
    '''
    功能: 两个变量X与Y（可以多个）的分析图。0-1离散型Y创建1的占比热力图；连续型Y创建均值热力图。本质上都是均值热力图。
    输入值: 
    X: 原始数据，dataframe类型
    cols_h: 水平轴选取字段，list类型，默认为data的所有列
    cols_v: 垂直轴选取字段，list类型，默认为data的所有列
    Y_cont: 连续型Y值，Series或一维np.array或DataFrame
    Y_cate: 0-1离散型Y值（暂时只能支持两类，且数值为0和1），Series或一维np.array或DataFrame
    feature_cate: 离散型X变量字段，list类型，默认为空
    quantiles: dict，键为变量名，值为list或一维数组，用于指定连续变量离散化的分位点，默认所有连续变量的分位点为[10*i for i in range(1,10)]
    cuts：dict，键为变量名，值为list或一维数组，用于直接指定连续变量离散化的分割点，优先级高于quantiles
    pattern: 正则表达式，用于匹配横轴标签字符串，使其按照该正则表达式提取后的数值排序
    str_nopattern: 字典，键为变量名（或变量位置），值为列表，表示未匹配pattern字符串的正常顺序
    fontsize: int，字体大小
    backend: 画图后端，可选{'seaborn','matplotlib'}
    close: 是否关闭生成的图
    show_last: 是否展示最后一幅图
    verbose: 是否打印日志。
    输出值: 
    fig_dict: X～Y关系图字典；键为二元组，第一个元素为水平轴字段名，第二个元素为垂直轴字段名，如('x1','x2')；值为热力图对象
    '''
    data=X.copy()
    if cols_v is None:
        cols_v=list(data.columns)
    if cols_h is None:
        cols_h=list(data.columns)
    if feature_cate is None:
        feature_cate=[]
    if quantiles is None:
        quantiles={}
    if cuts is None:
        cuts={}
    if str_nopattern is None:
        str_nopattern={}
    for key in quantiles:
        quantiles[key]=np.sort(np.unique(quantiles[key])).tolist()
    for key in cuts:
        cuts[key]=np.sort(np.unique(cuts[key]))
    q_default=[10*i for i in range(1,10)]
    #先对连续型变量离散化
    feature_cont=[col for col in cols_v+cols_h if col not in feature_cate]
    if len(feature_cont)>0:
        for column in feature_cont:
            clf = discretize.QuantileDiscretizer(quantiles=quantiles.get(column, q_default), return_numeric=False,
                                                 fill_na='Missing')
            if column in cuts.keys():
                clf.cuts = cuts[column]
            else:
                clf.fit(data[column])
            data[column] = clf.transform(data[column])
    data=data.fillna('Missing')
    if (Y_cont is None) and (Y_cate is None):
        raise Exception('Y值未给定！')
    if (Y_cont is None) and (Y_cate is None):
        raise Exception('连续型和离散型Y值只能给定一种！')
    if Y_cate is not None:
        Y=pd.DataFrame(Y_cate)
    else:
        Y=pd.DataFrame(Y_cont)
    fig_dict = {}
    n=Y.shape[1]
    cols_Y=list(Y.columns)
    cols_Y.sort()
    for vcol in cols_v:
        for hcol in cols_h:
            if verbose:
                print(vcol,hcol)
            if (vcol==hcol) or (vcol,hcol) in fig_dict.keys():
                continue
            fig,axes=plt.subplots(n,1,figsize=figsize)
            if n==1:
                axes=np.array([axes])
            for i,col in enumerate(cols_Y):
                value=Y[col].groupby([data[hcol],data[vcol]]).mean().unstack(hcol)
                if backend=='seaborn':
                    value=value.reindex_axis(utils.sort(value.index.tolist(),ascending=False,pattern=pattern,
                                                        str_nopattern=str_nopattern.get(vcol,None),converter=float),axis=0)
                else:
                    value=value.reindex_axis(utils.sort(value.index.tolist(),ascending=True,pattern=pattern,
                                                        str_nopattern=str_nopattern.get(vcol,None),converter=float),axis=0)
                value=value.reindex_axis(utils.sort(value.columns.tolist(),ascending=True,pattern=pattern,
                                                    str_nopattern=str_nopattern.get(hcol,None),converter=float),axis=1)
                value=value.fillna(0)
                if i==0:
                    title='Horizontal: %s <---> Vertical: %s\n%s'%(hcol,vcol,col)
                else:
                    title=col
                if backend=='seaborn':
                    if Y_cate is not None:
                        sns.heatmap(value,ax=axes[i],annot=True,fmt='.2%')
                    else:
                        sns.heatmap(value,ax=axes[i],annot=True,fmt='g')
                    axes[i].set_title(title,fontsize=fontsize)
                    axes[i].set_xlabel('')
                    axes[i].set_ylabel('')
                else:
                    pc,_=heatmap(value,ax=axes[i],
                                 xlabel='',ylabel='',
                                 xticklabels=value.columns,yticklabels=value.index,
                                 title=title,fontsize=fontsize)
            if backend!='seaborn':
                plt.colorbar(pc,ax=axes.ravel().tolist())
            plt.xticks(rotation=30)
            plt.yticks(rotation=30)
            fig_dict[(hcol,vcol)]=fig
            if close:
                plt.close('all')
    if show_last:
        try:
            fig
            fig.show()
        except:
            pass
    return fig_dict


#==============================================================================
# 测试
#==============================================================================

def test():
    np.random.seed(13)
    X=pd.DataFrame(np.random.choice([1,2,3],[300,3],p=[0.6,0.3,0.1]))
    X.columns=['x%d'%i for i in range(3)]
    X['x3']=np.random.choice(['a','b','c'],300)
    Y=pd.DataFrame(np.random.choice([0,1],[300,3],p=[0.9,0.1]))
    Y.columns=['y%d'%i for i in range(3)]
    
    
    result=plot_singleXY_PercentInY(X=X,Y_cate=Y['y1'],cols=None,feature_cate=list(X.columns),
                                    close=False,color_map={0:'g',1:'k'},legend_map={0:'good',1:'bad'},
                                    str_nopattern={'x3':['c','a','b']})
    
    result=plot_singleXY_Mean(X=X,Y=Y,cols=None,feature_cate=list(X.columns),normalize=False,close=False)
    
    result=plot_doubleXY_Mean(X=X,Y_cate=Y,feature_cate=list(X.columns),backend='matplotlib',close=False)

