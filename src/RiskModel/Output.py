# -*- coding: utf-8 -*-
"""
风险模型输出模块。
主要包括：
    最大减分项（线性模型）
    指标得分+最大减分项（knn模型）
    雷达图（线性模型）
    分数映射转化
    分数映射表（包括坏账率、加权坏账率）

"""

import pandas as pd
import numpy as np
import copy
from sklearn.neighbors import KNeighborsRegressor


class KNNExplain(object):
    '''
    利用knn解释模型中各个变量的影响度（针对每个样本），因此fit时给定的得分应该是模型得分（而不是真实得分）。
    方法如下：
        对于每个连续型变量，建立单变量knn回归模型，然后根据单变量的取值预测该样本的模型得分，作为该样本该变量的影响度；
        对于每个离散型变量，统计其不同取值下模型得分的均值分布，然后根据单变量的取值直接映射出该样本的模型得分。

    属性：
    feature_cate: 列表，表示离散型变量的名称或位置，默认为空。
    n_neighbors: int，knn中的k。
    verbose: bool，是否打印日志
    models: dict，键为变量名称或位置，值有两种可能：
            如果是离散型变量，则为字典（其键为变量取值，值为对应的模型得分均值）；
            如果是连续型变量，则为训练好的KNeighborsRegressor实例。
    means: float，整体的模型得分均值。
    '''
    def __init__(self,feature_cate=None,n_neighbors=5):
        if feature_cate is None:
            feature_cate=[]
        self.feature_cate=feature_cate
        self.n_neighbors=n_neighbors

    def fit(self,X,y):
        '''
        训练knn解释模型。
        :param X: 二维数组或者dataframe，表示训练样本各个变量的取值。
        :param y: 一维数组或者series，与X的行记录对应，表示模型得分（而不是真实得分）。
        :return:
        '''
        X=pd.DataFrame(X.copy())
        X=X.reset_index(drop=True)
        y=pd.Series(y.copy())
        y=y.reset_index(drop=True)
        self.means=y.mean()
        self.models={}
        for col in X.columns.tolist():
            if col in self.feature_cate:
                self.models[col]=y.groupby(X[col]).mean().to_dict()
            else:
                knn=KNeighborsRegressor(n_neighbors=self.n_neighbors)
                knn.fit(X[[col]],y)
                self.models[col]=copy.deepcopy(knn)
        return self

    def explain(self,X):
        '''
        利用训练好的knn解释模型计算每个样本每个变量的影响度。
        :param X: 二维数组或者dataframe。
        :return: 二维数组或者dataframe，表示每个样本每个变量的影响度。
        '''
        if isinstance(X,pd.DataFrame):
            isdf=True
        else:
            isdf=False
        X=pd.DataFrame(X.copy())
        result=pd.DataFrame()
        result=result.reindex(index=X.index)
        for col in X.columns.tolist():
            if col not in self.models:
                continue
            else:
                if isinstance(self.models[col],dict):
                    result[col]=X[col].map(lambda xx: self.models[col].get(xx,self.means))
                else:
                    result[col]=self.models[col].predict(X[[col]])
        if isdf:
            return result
        else:
            return result.values





def GetTopImp(X, coefs, top=5, normalize=(0, 5)):
    '''
    获取线性模型的前top个最大减分项，并归一化到normalize区间内。
    目前的方法是将变量取值与变量系数相乘，然后获取得分为负的，abs后归一化，截取前top个。
    注意：变量取值最好是标准化后的，变量系数最好也是变量标准化后回归模型的系数，否则可能出现没有负分的情况。
         如果没有负分项，目前的处理方式是返回空列表；如果负分项不足top个，则返回的列表也会不足top项。
         如果逻辑回归模型定义的Y=1为坏人，而最大减分项针对的是好人的减分项，则输入方面需要处理一下，即对X或者coefs先取负号。
    :param X: series，其index为变量名，值为变量取值，代表一个客户。
    :param coefs: 列表或一维数组或series，表示各变量对应的系数，
                  若为列表或一维数组，顺序应与X的顺序一致；若为series，其index应为变量名。
    :param top: int，获取前top个最大减分项。
    :param normalize: 二元tuple，表示归一化区间上下限。
    :return: 列表，每个元素为二元tuple（分别为变量名和归一化取值），且按顺序从高到低排列（减分作用）。
    '''
    index = X.index.tolist()
    if isinstance(coefs, pd.Series):
        coefs = coefs.reindex(index=index)
    else:
        coefs = pd.Series(coefs)
        coefs.index = index
    result = X * coefs
    result = result.loc[result < 0]
    if result.shape[0] == 0:
        return []
    result = result.abs().sort_values(ascending=False)
    r_min = result.min()
    r_max = result.max()
    if r_min == r_max:
        return [(xx[0], normalize[1]) for xx in result.head(top).items()]
    result = (result - r_min) / float(r_max - r_min)
    result = normalize[0] + result * (normalize[1] - normalize[0])
    return list(result.head(top).items())


def GetRadar(X, coefs_radar, maps_from, maps_to, score_min=None, score_max=None, normalize=(0, 5)):
    '''
    雷达图数据计算。
    根据线性模型计算出每个维度的得分，再根据对应的分割点获取每个维度的得分映射，并归一化到normalize区间内。
    :param X: series，其index为变量名，值为变量取值，代表一个客户。
    :param coefs_radar: dataframe，其index为变量名，columns为['coefs','radar']，分别表示变量系数及其所属维度。
    :param maps_from: dict of list，键为雷达图维度编号（或名称），值为原始得分分割点list，升序排列。
    :param maps_to: dict of list，键为雷达图维度编号（或名称），值为转化得分分割点list，与maps_from对应，须为0到100之间，升序排列。
    :param score_min: int，原始得分空间最小值，用于处理端点情况，None则使用指数衰减。
    :param score_max: int，原始得分空间最大值，用于处理端点情况，None则使用指数衰减。
    :param normalize: 二元tuple，表示归一化区间上下限。
    :return: 字典，键为雷达图维度编号（或名称），值为维度得分。
    '''
    coefs_radar['score'] = X * coefs_radar['coefs']
    scores = coefs_radar.groupby('radar')['score'].sum().to_dict()
    results = {key: map_util(scores[key], maps_from[key], maps_to[key], score_min=score_min, score_max=score_max,
                             normalize=normalize) for key in scores}
    return results


def map_util(score, maps_from, maps_to, score_min=None, score_max=None, normalize=None):
    '''
    根据原始得分分割点（maps_from）和转化得分分割点（maps_to）的映射关系转化得分（score）。
    作用：可以用于辅助概率分转化为信用得分、人群占比排名计算。

    例子：
    maps_from=[8,12,45,67,157]  #原始得分分割点
    maps_to=[5,10,20,50,100]    #转化得分分割点，每一项与maps_from对应，如原始得分8对应的转化得分为5，原始得分157对应的转化得分为100
    实际操作是将score从maps_from空间映射到maps_to空间，中间取值采取线性插值的方式。
    目前需要限制maps_to空间为0到100，方便处理端点取值的情况，然后根据normalize映射到对应区间。

    :param score: int，原始得分（或概率值或字段取值均可）。
    :param maps_from: list，原始得分分割点，升序排列。
    :param maps_to: list，转化得分分割点，与maps_from对应，须为0到100之间，升序排列。
    :param score_min: int，原始得分空间最小值，用于处理端点情况，None则使用指数衰减。
    :param score_max: int，原始得分空间最大值，用于处理端点情况，None则使用指数衰减。
    :param normalize: 二元tuple，表示归一化区间上下限，默认为(0,100)。
    :return: int，转化得分。
    '''
    if normalize is None:
        normalize = (0, 100)
    low = np.searchsorted(maps_from, score)
    diff_max = np.diff(maps_from).max()
    if low == 0:
        if maps_to[0] == 0:
            result = 0
        elif (score_min is None) or (score_min >= maps_from[0]):
            result = maps_to[0] * np.exp((score - maps_from[0]) / float(diff_max))
        elif score < score_min:
            result = 0
        else:
            result = (score - score_min) * maps_to[0] / float(maps_from[0] - score_min)
    elif low == len(maps_from):
        if maps_to[-1] == 100:
            result = 100
        elif (score_max is None) or (score_max <= maps_from[-1]):
            result = 100 - (100 - maps_to[-1]) * np.exp((maps_from[0] - score) / float(diff_max))
        elif score > score_max:
            result = 100
        else:
            result = maps_to[-1] + (score - maps_from[-1]) * (100 - maps_to[-1]) / float(score_max - maps_from[-1])
    else:
        result = maps_to[low - 1] + (score - maps_from[low - 1]) * (maps_to[low] - maps_to[low - 1]) / float(
            maps_from[low] - maps_from[low - 1])
    return normalize[0] + (normalize[1] - normalize[0]) * result / float(100)


def ScoreMapsSummary(data, score_cuts=None, bad_label=1):
    '''
    获取分数映射表（包括区间坏账率、累计坏账率等）。
    data: dataframe，包含两列（score、isbad）或三列（score、isbad、weight），如果是两列（不包含weight），则默认每个样本的权重为1。
    score_cuts: 列表，表示分数分割的上下限，默认为list(range(0,105,5))。
    bad_label: 违约人群的标签。
    返回dataframe：
        分数下限
        分数上限
        人数
        人数区间占比
        人数累计占比
        人数区间加权占比
        人数累计加权占比
        违约人数
        区间坏账率
        累计坏账率
        区间加权坏账率
        累计加权坏账率
    '''
    newdata = data.copy()
    newdata['isbad'] = (newdata['isbad'] == bad_label).astype(float)
    if score_cuts is None:
        score_cuts = list(range(0, 105, 5))
    score_cuts.sort()
    if 'weight' not in newdata.columns.tolist():
        newdata['weight'] = 1
    result = pd.DataFrame()
    result['分数下限'] = score_cuts[:-1]
    result['分数上限'] = score_cuts[1:]
    result.index = range(result.shape[0])
    result = result.sort_values('分数下限', ascending=False)
    newdata['score'] = np.searchsorted(score_cuts[1:-1], newdata['score'])

    total = float(newdata.shape[0])
    total_weighted = float(newdata['weight'].sum())
    result['人数'] = newdata.groupby('score')['isbad'].count()
    result['加权人数'] = newdata.groupby('score')['weight'].sum()
    result['人数区间占比'] = result['人数'] / total
    result['人数累计占比'] = result['人数区间占比'].cumsum()
    result['人数区间加权占比'] = result['加权人数'] / total_weighted
    result['人数累计加权占比'] = result['人数区间加权占比'].cumsum()
    result['违约人数'] = newdata.groupby('score')['isbad'].sum()
    result['区间坏账率'] = result['违约人数'] / result['人数']
    result['累计坏账率'] = result['违约人数'].cumsum() / result['人数'].cumsum()

    newdata['isbad'] = newdata['isbad'] * newdata['weight']
    result['加权违约人数'] = newdata.groupby('score')['isbad'].sum()
    result['区间加权坏账率'] = result['加权违约人数'] / result['加权人数']
    result['累计加权坏账率'] = result['加权违约人数'].cumsum() / result['加权人数'].cumsum()

    result = result.drop(['加权人数', '加权违约人数'], axis=1)
    columns_out = ['分数下限', '分数上限', '人数', '人数区间占比', '人数累计占比', '人数区间加权占比',
                   '人数累计加权占比', '违约人数', '区间坏账率', '累计坏账率', '区间加权坏账率', '累计加权坏账率']
    result = result[columns_out]
    return result


def test():
    np.random.seed(13)
    X = pd.Series(np.random.randn(20))
    X.index = ['x%d' % i for i in range(X.shape[0])]

    coefs_radar = pd.DataFrame()
    coefs_radar['coefs'] = 2 * np.random.random(X.shape[0]) - 1
    coefs_radar['radar'] = np.random.choice(['A', 'B', 'C'], X.shape[0])
    coefs_radar.index = X.index

    # 提取减分最多的几项，并归一化
    GetTopImp(X, coefs_radar['coefs'], top=5, normalize=(0, 5))

    # map_util测试（通过设置参数可直接进行分数映射、人群占比计算）
    map_util(score=34, maps_from=[8, 12, 45, 67, 157], maps_to=[5, 10, 20, 50, 100], score_min=None, score_max=None)
    map_util(score=67, maps_from=[8, 12, 45, 67, 157], maps_to=[5, 10, 20, 50, 100], score_min=None, score_max=None)
    map_util(score=157, maps_from=[8, 12, 45, 67, 157], maps_to=[5, 10, 20, 50, 100], score_min=None, score_max=None)
    map_util(score=160, maps_from=[8, 12, 45, 67, 157], maps_to=[5, 10, 20, 50, 100], score_min=None, score_max=200)
    map_util(score=-100, maps_from=[8, 12, 45, 67, 157], maps_to=[5, 10, 20, 50, 100], score_min=None, score_max=None)
    map_util(score=-100, maps_from=[8, 12, 45, 67, 157], maps_to=[5, 10, 20, 50, 100], score_min=0, score_max=None)

    # 提取雷达图维度得分
    GetRadar(X, coefs_radar, maps_from={'A': [-2, -1, 0, 1, 2], 'B': [-1, 0, 1], 'C': [-1, 1]},
             maps_to={'A': [1, 30, 50, 70, 98], 'B': [10, 50, 90], 'C': [25, 75]}, score_min=None, score_max=None,
             normalize=(0, 5))

    # 获取分数映射表
    data = pd.DataFrame()
    data['score'] = np.random.rand(1000) * 100
    data['isbad'] = np.random.choice([0, 1], 1000)
    ScoreMapsSummary(data, score_cuts=None, bad_label=1)

    # knn解释模型
    data = pd.read_csv('data/credit_card_data.csv', encoding='utf8', index_col=[0])
    y = data['SeriousDlqin2yrs'].copy()
    X = data[[col for col in data.columns.tolist() if col != 'SeriousDlqin2yrs']].copy()
    X=X.fillna(X.mean())
    from sklearn.linear_model import LogisticRegression
    lr=LogisticRegression()
    lr.fit(X,y)
    y_model=pd.Series(lr.predict_proba(X)[:,1],index=X.index)

    clf_explain=KNNExplain(feature_cate=None,n_neighbors=5,verbose=True)
    clf_explain.fit(X,y_model)
    explain_knn=clf_explain.explain(X)

    GetTopImp(X.iloc[0], lr.coef_.squeeze().tolist(), top=5, normalize=(0, 5))


def kaiyuanjinrong():
    weight = pd.DataFrame()
    weight['授信日期'] = ['201512', '201601', '201602', '201603', '201604', '201605', '201606']
    weight['违约总人数'] = [324, 244, 167, 45, 12, 74, 104]
    weight['未违约总人数'] = [1345, 958, 751, 449, 207, 638, 603]
    weight['违约抽样人数'] = [324, 244, 167, 45, 12, 74, 104]
    weight['未违约抽样人数'] = [1345, 759, 580, 312, 138, 450, 446]

    weight_map_bad = dict(zip(weight['授信日期'], weight['违约总人数'] / weight['违约抽样人数']))
    weight_map_good = dict(zip(weight['授信日期'], weight['未违约总人数'] / weight['未违约抽样人数']))

    data = pd.read_csv('data/scores_20170517.csv')
    data['授信日期'] = pd.to_datetime(data['授信日期']).map(lambda xx: xx.strftime('%Y%m'))
    data['weight'] = 1
    data.loc[data['Y'] == 0, 'weight'] = data.loc[data['Y'] == 0, '授信日期'].map(lambda xx: weight_map_good[xx])
    data.loc[data['Y'] == 1, 'weight'] = data.loc[data['Y'] == 1, '授信日期'].map(lambda xx: weight_map_bad[xx])

    newdata = pd.DataFrame()
    newdata['score'] = (1 - data['消费分期分数']) * 100
    newdata['isbad'] = data['Y']
    newdata['weight'] = data['weight']
    result = ScoreMapsSummary(newdata, score_cuts=list(range(0, 105, 5)), bad_label=1)

    newdata = pd.DataFrame()
    newdata['score'] = (1 - data['卡车信贷分数']) * 100
    newdata['isbad'] = data['Y']
    newdata['weight'] = data['weight']
    result = ScoreMapsSummary(newdata, score_cuts=list(range(0, 105, 5)), bad_label=1)









