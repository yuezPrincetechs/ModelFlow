# -*- coding: utf-8 -*-
'''
mysql辅助函数。
'''

from sqlalchemy import create_engine
import sqlalchemy
import pandas as pd
import numpy as np

class MysqlHelper(object):

    def __init__(self,ip,user,password,db,port=3306,charset='utf8mb4',encoding='utf-8',driver='mysql+pymysql'):
        '''
        数据库连接初始化。
        不同的driver需要安装不同的包，如pymysql。
        :param ip:
        :param user:
        :param password:
        :param db:
        :param port:
        :param charset:
        :param encoding:
        :param driver:
        :return:
        '''
        #self.url='mysql+pymysql://%s:%s@%s:%s/%s?charset=%s'%(user,password,ip,port,db,charset)
        #下面的方式更为通用，不易出错
        self.driver=driver
        self.ip=ip
        self.user=user
        self.password=password
        self.db=db
        self.port=port
        self.charset=charset
        self.encoding=encoding
        self.url=self.get_url()
        self.engine=create_engine(self.url,encoding=self.encoding)

    def get_url(self):
        return sqlalchemy.engine.url.URL(drivername=self.driver, username=self.user, password=self.password, host=self.ip, port=self.port, database=self.db,query=dict(charset=self.charset))

    def execute(self,statement):
        '''
        执行sql语句。
        :param statement:
        :return:
        '''
        self.engine.execute(statement)

    def to_sql(self,data,table_name,delete=False,if_exists='append',index=None,chunksize=5000):
        '''
        数据传输到数据库对应表格中。
        :param data: 数据框。
        :param table_name: 数据库对应表格名称。
        :param delete: 是否清空表格数据。
        :param if_exists:
        :param index:
        :param chunksize:
        :return:
        '''
        if delete is True:
            self.execute('TRUNCATE TABLE %s;'%table_name)
        data.to_sql(table_name,self.engine,if_exists=if_exists,index=index,chunksize=chunksize)

    def read_table(self,table_name,cols=None,chunksize=None,return_generator=True):
        '''
        读取数据库表格数据。
        :param table_name: 数据库对应表格名称。
        :param cols: 需要读取的变量列表，默认None表示全部读取。
        :param chunksize: int，分块读取数据时每次读取的记录数，默认一次性读取。
        :param return_generator: bool，是否返回生成器，False则会将所有数据拼接后返回。（仅当chunksize为int时才有效）
        :return:
        '''
        if cols is None:
            cols=['*']
        sql='select %s from %s;'%(','.join(cols),table_name)
        result=self.read_sql(sql,chunksize=chunksize,return_generator=return_generator)
        return result

    def read_sql(self,sql,chunksize=None,return_generator=True):
        '''
        根据语句读取数据。
        :param sql: sql语句。
        :param chunksize: int，分块读取数据时每次读取的记录数，默认一次性读取。
        :param return_generator: bool，是否返回生成器，False则会将所有数据拼接后返回。（仅当chunksize为int时才有效）
        :return:
        '''
        if chunksize is not None and chunksize<=0:
            chunksize=None
        result=pd.read_sql(sql,self.engine,chunksize=chunksize)
        if return_generator:
            return result
        else:
            if chunksize is None:
                return result
            else:
                result=list(result)
                if len(result)==0:
                    return pd.DataFrame()
                else:
                    result=pd.concat(result,axis=0)
                    return result

    def update_existed(self,t1,t2,cols_set,cols_fixed):
        '''
        根据表格t2的数据更新表格t1，需要更新的字段列表为cols_set，需要固定的字段列表为cols_fixed。
        只更新cols_fixed中所有字段都相同的记录。
        :param t1: 需要更新的表格名称。
        :param t2: 用于更新的表格名称（相当于临时表）。
        :param cols_set: 需要更新的字段列表。
        :param cols_fixed: 需要固定的字段列表。
        :return:
        '''
        sql_on=' and '.join(['%s.%s=%s.%s'%(t1,col,t2,col) for col in cols_fixed])
        sql_set=','.join(['%s.%s=%s.%s'%(t1,col,t2,col) for col in cols_set])
        sql='update %s inner join %s on %s set %s;'%(t1,t2,sql_on,sql_set)
        self.execute(sql)

    def update_noexisted(self,t1,t2,cols_select=None,set_unique=False,cols_unique=None):
        '''
        根据表格t2的数据更新表格t1，根据unique索引只插入不重复的记录（忽略已存在的记录）。
        :param t1: 需要更新的表格名称。
        :param t2: 用于更新的表格名称（相当于临时表）。
        :param cols_select: 需要更新插入的字段列表，默认None表示全部字段。
        :param set_unique: 是否设定unique索引，如果为True，cols_unique不能为None。
        :param cols_unique: 需要设定unique索引的字段列表，默认值为None。
        :return:
        '''
        if set_unique is True:
            if cols_unique is None:
                raise Exception('cols_unique must be given when set_unique is True!')
            else:
                sql_addunique='alter table %s add unique (%s);'%(t1,','.join(cols_unique))
                self.execute(sql_addunique)
        if cols_select is None:
            cols_select=['*']
        sql_update='insert ignore into %s select %s from %s;'%(t1,','.join(cols_select),t2)
        self.execute(sql_update)

def update_sql(data,sql_model,table_name,cols_fixed,cols_set=None,set_unique=False):
    '''
    将数据更新到数据库中，数据库中的表需要设置唯一键。
    :param data: 数据框，所有数据类型符合数据库中表的字段类型。
    :param sql_model: MysqlHelper对象实例。
    :param table_name: 需要更新的表名。
    :param cols_fixed: 需要固定的字段列表（需要提前设置唯一键）。
    :param cols_set: 需要更新的字段列表（默认全部更新），若为空列表，则不更新已有数据，仅插入新数据。
    :param set_unique: 是否设定unique索引。
    :return:
    '''
    table_tmp=table_name+'_tmp'
    sql_model.execute('drop table if exists %s;'%table_tmp)
    sql_model.execute('create table %s like %s;'%(table_tmp,table_name))
    if cols_set is None:
        cols_set=[xx for xx in data.columns if xx not in cols_fixed]
    sql_model.to_sql(data,table_tmp,delete=True,if_exists='append',index=None)
    if len(cols_set)>0:
        sql_model.update_existed(t1=table_name,t2=table_tmp,cols_set=cols_set,cols_fixed=cols_fixed)
    sql_model.update_noexisted(t1=table_name,t2=table_tmp,set_unique=set_unique,cols_unique=cols_fixed)
    sql_model.execute('drop table if exists %s;'%table_tmp)

