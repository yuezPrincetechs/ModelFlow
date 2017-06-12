# -*- coding: utf-8 -*-
'''
一些辅助函数。
'''

import pandas as pd
import numpy as np
import re
import os
import errno
import hashlib

def sort(l,ascending=True,pattern='\((.*?),',str_nopattern=None,converter=float):
    '''
    根据pattern正则表达式提取的信息进行排序，排序顺序为: [pattern匹配排序结果,未匹配字符串排序结果,数值型排序结果]。
    l: 需要排序的列表。
    ascending: True表示升序，False表示降序。
    pattern: 匹配正则表达式，None则表示不需要正则表达式匹配信息。
    str_nopattern: 列表，表示未匹配字符串的正常顺序。
    converter: 正则表达式匹配得到的信息转换函数，None则表示不需要转换。
    返回排序后的列表。
    '''
    l_str=[i for i in l if isinstance(i,str)]
    l_numeric=sorted([i for i in l if i not in l_str],reverse=not ascending)
    if len(l_str)==0:
        return l_numeric
    l_str_pattern=[]
    l_str_nopattern=[]
    if pattern is None:
        return sorted(l_str,reverse=not ascending)+l_numeric
    for i in l_str:
        g=re.findall(pattern,i)
        if len(g)==0:
            l_str_nopattern.append(i)
            continue
        if converter is None:
            l_str_pattern.append((i,g[0]))
        else:
            l_str_pattern.append((i,converter(g[0])))
    if str_nopattern is None:
        l_str_nopattern=sorted(l_str_nopattern,reverse=not ascending)
    else:
        l_str_nopattern_exist=[i for i in l_str_nopattern if i in l_str_nopattern]
        l_str_nopattern_noexist=[i for i in l_str_nopattern if i not in l_str_nopattern]
        l_str_nopattern_exist=sorted(l_str_nopattern_exist,reverse=not ascending,key=lambda xx: str_nopattern.index(xx))
        l_str_nopattern_noexist=sorted(l_str_nopattern_noexist,reverse=not ascending)
        l_str_nopattern=l_str_nopattern_exist+l_str_nopattern_noexist
    l_str_pattern=sorted(l_str_pattern,reverse=not ascending,key=lambda xx: xx[1])
    l_str_pattern=[xx[0] for xx in l_str_pattern]
    return l_str_pattern+l_str_nopattern+l_numeric

def delFromList(features,features_todel):
    '''
    从已有列表中删除指定元素。
    features: 已有元素列表。
    features_todel: 需要删除的元素列表。
    '''
    return [i for i in features if i not in features_todel]

def addToList(features,features_toadd):
    '''
    从已有列表中添加指定元素。
    features: 已有元素列表。
    features_toadd: 需要添加的元素列表。
    '''
    result=[]
    result.extend(features)
    for i in features_toadd:
        if i in result:
            pass
        else:
            result.append(i)
    return result


def ensure_directory(directory):
    '''
    判断路径（一般是文件夹路径）是否存在，若不存在，则会自动创建（包括不存在的上层路径）。
    directory: 文件夹路径名称。
    '''
    directory=os.path.expanduser(directory)
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno!=errno.EEXIST:
            raise e

def clear_directory(directory,self_delete=True):
    '''
    递归清空文件夹／文件。
    :param directory: 字符串，文件夹／文件路径名称。
    :param self_delete: 布尔值，是否删除directory本身。
    :return:
    '''
    directory=os.path.expanduser(directory)
    if not os.path.exists(directory):
        return None
    if os.path.isfile(directory):
        if self_delete:
            os.remove(directory)
    else:
        for item in os.listdir(directory):
            item_path=os.path.join(directory,item)
            clear_directory(item_path,self_delete=True)
        if self_delete:
            os.rmdir(directory)

def myhash(s):
    '''
    对字符串进行md5加密，并转换为十进制。
    '''
    return int(hashlib.md5(s.encode('utf-8')).hexdigest(),16)


def splitByID(infile,ID_list,outfile=None,ngroup=10,columns_raw=None,columns_new=None,sep=',',chunksize=100000,encoding='utf-8',dtype=str):
    '''
    根据唯一ID分割文件，采用md5加密哈希的方法使得各个组文件大小尽量平均。（需要使用hashlib，python自带hash每次结果不一样，非常不安全）
    
    infile: 字符串，需要处理的数据文件名，以.csv结尾。
    ID_list: 列表，分割文件依据的主键ID，多个时与顺序有关。
    outfile: 字符串，生成的文件名，以.csv结尾，输出时会追加组号，默认为infile。
    ngroup: 正整数，需要划分的组数，应大于1。
    columns_raw: 字符串列表，列名信息，None则会将数据第一行当作columns，否则会将数据第一行也当作数据项。
    columns_new: 字符串列表，保存成文件时需要的列，None则表示全部保存。
    sep: 字符串，读入文件的分隔符，输出一律使用逗号分隔。
    chunksize: 正整数，每次处理的数据量。
    encoding: 字符串，读入文件的编码格式，为了统一编码，输出一律用utf-8。
    dtype: 参考pandas.read_csv的dtype参数。
    '''
    if outfile is None:
        outfile=infile
    if len(sep)==1:
        if columns_raw is None:
            data_iter=pd.read_csv(infile,header='infer',encoding=encoding,sep=sep,dtype=dtype,chunksize=chunksize)
        else:
            data_iter=pd.read_csv(infile,header=None,encoding=encoding,sep=sep,dtype=dtype,chunksize=chunksize)
        for i,data in enumerate(data_iter):
            if i==0 and columns_raw is None:
                columns_raw=data.columns.tolist()
            if i==0 and columns_new is None:
                columns_new=columns_raw.copy()
            data.columns=columns_raw
            data[ID_list]=data[ID_list].fillna('')
            if len(ID_list)==1:
                data_group=data[ID_list[0]].map(lambda xx: divmod(myhash(xx),ngroup)[1])
            else:
                data_group=data[ID_list].apply(lambda xx: ''.join(xx),axis=1).map(lambda xx: divmod(myhash(xx),ngroup)[1])
            for g in data.groupby(data_group):
                # g[0]为组别号，g[1]为对应数据
                filename=outfile.replace('.csv','_%d.csv'%g[0])
                if os.path.exists(filename):
                    g[1].to_csv(filename,mode='a',header=None,
                                index=None,columns=columns_new,encoding='utf-8',sep=',')
                else:
                    g[1].to_csv(filename,mode='w',header=True,
                                index=None,columns=columns_new,encoding='utf-8',sep=',')
            print('Done:',i)
    else:
        data=[]
        j=0
        for i,line in enumerate(open(infile,encoding=encoding)):
            line=line.split(sep)
            line[-1]=line[-1].strip()
            if i==0:
                if columns_raw is None:
                    columns_raw=line
                else:
                    data.append(line.copy())
                if columns_new is None:
                    columns_new=columns_raw.copy()
                continue
            data.append(line.copy())
            if i<(j+1)*chunksize:
                continue
            else:
                data=pd.DataFrame(data)
                data.columns=columns_raw
                data[ID_list]=data[ID_list].fillna('')
                if len(ID_list)==1:
                    data_group=data[ID_list[0]].map(lambda xx: divmod(myhash(xx),ngroup)[1])
                else:
                    data_group=data[ID_list].apply(lambda xx: ''.join(xx),axis=1).map(lambda xx: divmod(myhash(xx),ngroup)[1])
                for g in data.groupby(data_group):
                    # g[0]为组别号，g[1]为对应数据
                    filename=outfile.replace('.csv','_%d.csv'%g[0])
                    if os.path.exists(filename):
                        g[1].to_csv(filename,mode='a',header=None,
                                    index=None,columns=columns_new,encoding='utf-8',sep=',')
                    else:
                        g[1].to_csv(filename,mode='w',header=True,
                                    index=None,columns=columns_new,encoding='utf-8',sep=',')
                print('Done:',j)
                j+=1
                data=[]
        if len(data)>0:
            data=pd.DataFrame(data)
            data.columns=columns_raw
            data[ID_list]=data[ID_list].fillna('')
            if len(ID_list)==1:
                data_group=data[ID_list[0]].map(lambda xx: divmod(myhash(xx),ngroup)[1])
            else:
                data_group=data[ID_list].apply(lambda xx: ''.join(xx),axis=1).map(lambda xx: divmod(myhash(xx),ngroup)[1])
            for g in data.groupby(data_group):
                # g[0]为组别号，g[1]为对应数据
                filename=outfile.replace('.csv','_%d.csv'%g[0])
                if os.path.exists(filename):
                    g[1].to_csv(filename,mode='a',header=None,
                                index=None,columns=columns_new,encoding='utf-8',sep=',')
                else:
                    g[1].to_csv(filename,mode='w',header=True,
                                index=None,columns=columns_new,encoding='utf-8',sep=',')
            print('Done:',j)



