# -*- coding: utf-8 -*-
'''
日志记录，可同时将标准输出与标准错误输出到文件和控制台。
自带logging模块需要更改所有的print语句，此自定义模块不需要对原始程序做任何更改。
网上看应该也可以用logging改写，待研究。
'''

import sys, os

class Logger(object):
    '''
    日志记录类，支持同时将标准输出和标准错误输出到文件和控制台。
    注意：不能使用with语句，否则错误信息无法输出到文件。

    参数：
    tofile_out: 布尔值，是否将stdout输出到文件。
    tofile_error: 布尔值，是否将stderr输出到文件。
    toconsole_out: 布尔值，是否stdout输出到控制台。
    toconsole_error: 布尔值，是否将stderr输出到控制台。
    filename: 字符串，日志文件路径，若tofile_out和tofile_error均为False，则不需要提供。
    mode: 字符串，文件打开模式。
    kwds: 其他参数，见open。

    属性：
    log_out: 用于记录标准输出的LoggerHelper实例对象。
    log_error: 用于记录标准错误的LoggerHelper实例对象。
    '''
    def __init__(self, tofile_out=False, tofile_error=False, toconsole_out=True, toconsole_error=True,filename=None, mode="a", **kwds):
        self.log_out=LoggerHelper(message_type='stdout', tofile=tofile_out, toconsole=toconsole_out, filename=filename, mode=mode, **kwds)
        self.log_error=LoggerHelper(message_type='stderr', tofile=tofile_error, toconsole=toconsole_error, filename=filename, mode=mode, **kwds)

    def __del__(self):
        self.close()

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.log_out.close()
        self.log_error.close()


class LoggerHelper(object):
    '''
    日志记录类（辅助），支持同时将标准输出或者标准错误输出到文件和控制台。
    注意：不能使用with语句，否则错误信息无法输出到文件。

    参数：
    message_type: str，{'stdout','stderr'}，分别表示标准输出、标准错误。
    tofile: 布尔值，是否将信息输出到文件。
    toconsole: 布尔值，是否信息输出到控制台。
    filename: 字符串，日志文件路径，若tofile为False，则不需要提供。
    mode: 字符串，文件打开模式。
    kwds: 其他参数，见open。

    属性：
    std_message: 用于记录原始sys.stdout或者sys.stderr。
    file: open对象。
    '''
    def __init__(self, message_type='stdout', tofile=False, toconsole=True, filename=None, mode="a", **kwds):
        self.message_type=message_type
        self.tofile = tofile
        self.toconsole = toconsole
        if self.tofile:
            f = open(filename, mode=mode, **kwds)
            self.file = f
        else:
            self.file = None
        if self.message_type=='stdout':
            self.std_message=sys.stdout
            sys.stdout=self
        elif self.message_type=='stderr':
            self.std_message=sys.stderr
            sys.stderr=self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, message):
        if self.toconsole and (self.std_message is not None):
            self.std_message.write(message)
        if self.tofile and (self.file is not None):
            self.file.write(message)

    def flush(self):
        if self.std_message is not None:
            self.std_message.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        if self.std_message is not None:
            if self.message_type=='stdout':
                sys.stdout=self.std_message
            if self.message_type=='stderr':
                sys.stderr=self.std_message
        self.std_message=None
        if self.file is not None:
            self.file.close()
            self.file = None



def test():
    tofile_out=True
    tofile_error=True
    toconsole_out=False
    toconsole_error=True
    log=Logger(tofile_out=tofile_out, tofile_error=tofile_error, toconsole_out=toconsole_out, toconsole_error=toconsole_error,
               filename='/Users/yuez/Desktop/mylog.log', mode="a", encoding='utf-8')
    print('-'*30)
    print('tofile_out:',tofile_out,'tofile_error',tofile_error,'toconsole_out',toconsole_out,'toconsole_error',toconsole_error)
    print('this is a test of INFO!')
    print('Next is a test of ERROR:')
    print(1/0)
    log.close()