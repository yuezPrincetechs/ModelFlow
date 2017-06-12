# -*- coding: utf-8 -*-
'''
资源监控数据解析。
例如：Linux下使用命令'sar -r 1 > memory.txt &'可每隔一秒生成内存监控数据，本程序用于对这些监控文本数据
的简单解析，便于之后的统计分析、可视化等。
常用命令（具体可参考网上sar命令）：
sar -u 1 > cpu.txt &
sar -r 1 > memory.txt &
sar -b 1 > io.txt &
'''

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

def convert2float(xx):
    '''
    将xx转换为浮点型，若无法转换，则返回xx本身。
    :param xx:
    :return:
    '''
    try:
        return float(xx)
    except:
        return xx

def parse_raw(filepath,seconds=1):
    '''
    读取filepath中的原始监控数据，并解析。
    :param filepath: 字符串，原始监控数据存储路径。
    :param seconds: int，监控时间间隔，以秒为单位。
    :return: dataframe，其index为监控时间，columns为监控指标。
    '''
    data_head=pd.read_csv(filepath,delim_whitespace=True,header=None,nrows=1)
    data=pd.read_csv(filepath,delim_whitespace=True,header=None,skiprows=2)
    date_start=data_head.iloc[0,3]
    time_start=data.iloc[1,0]+' '+data.iloc[1,1]
    datetime_start=pd.to_datetime(date_start+' '+time_start)
    columns=list(data.iloc[0,2:])
    newdata=data.iloc[1:,2:].applymap(convert2float)
    newdata=newdata.dropna(axis=0,how='any')
    newdata=newdata.loc[(newdata.applymap(type)==type('')).sum(axis=1)<newdata.shape[1]]
    newdata=newdata.applymap(convert2float)
    newdata.columns=columns
    newdata.index=pd.date_range(start=datetime_start,periods=newdata.shape[0],freq='%dS'%seconds)
    newdata.index.name='datetime'
    return newdata

def parse_useful(filepath,seconds=1,index='memory'):
    '''
    读取filepath中的原始监控数据，并解析提取出常用信息。目前仅支持：内存、CPU、IO。
    :param filepath: 字符串，原始监控数据存储路径。
    :param seconds: int，监控时间间隔，以秒为单位。
    :param index: {'memory','cpu','io'}
    :return: dataframe，其index为监控时间，columns为监控常用指标。
    '''
    data=parse_raw(filepath=filepath,seconds=seconds)
    if index.lower()=='memory':
        data['MemUsed(MB)']=data['kbmemused']/1024.0
        data['MemUsed(%)']=data['%memused']
        data['Cached(MB)']=data['kbcached']/1024.0
        result=data[['MemUsed(MB)','MemUsed(%)','Cached(MB)']].copy()
    elif index.lower()=='cpu':
        data['cpuUsedByUser(%)']=data['%user']
        data['cpuUsedBySystem(%)']=data['%system']
        data['cpuUsedByIOwait(%)']=data['%iowait']
        result=data[['cpuUsedByUser(%)','cpuUsedBySystem(%)','cpuUsedByIOwait(%)']].copy()
    elif index.lower()=='io':
        result=data[['tps','rtps','wtps','bread/s','bwrtn/s']].copy()
    else:
        raise Exception('Unknown type: %s'%index)
    return result