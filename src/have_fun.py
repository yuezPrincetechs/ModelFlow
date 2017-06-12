import numpy as np


def quicksort(l,left=None,right=None):
    '''
    快速排序。
    :param l:
    :param left:
    :param right:
    :return:
    '''
    if left is None:
        left=0
    if right is None:
        right=len(l)-1
    if left>=right:
        return
    i=left
    j=right
    mid=int((i+j)/2.0)
    key=l[mid]
    l[mid]=l[i]
    while i<j:
        while l[j]>key and i<j:
            j-=1
        l[i]=l[j]
        while l[i]<=key and i<j:
            i+=1
        l[j]=l[i]
    l[i]=key
    quicksort(l,left,i-1)
    quicksort(l,i+1,right)

def merge(l1,l2):
    '''
    归并操作。
    :param l1:
    :param l2:
    :return:
    '''
    n1=len(l1)
    n2=len(l2)
    l=[]
    i=0
    j=0
    while i<n1 and j<n2:
        if l1[i]<=l2[j]:
            l.append(l1[i])
            i+=1
        else:
            l.append(l2[j])
            j+=1
    if i==n1:
        l.extend(l2[j:])
    else:
        l.extend(l1[i:])
    return l


def mergesort(l):
    '''
    归并排序。
    :param l:
    :return:
    '''
    if len(l)<=1:
        return l
    k=int(len(l)/2.0)
    left=mergesort(l[:k])
    right=mergesort(l[k:])
    result=merge(left,right)
    return result

def test():
    np.random.seed(13)
    l = np.random.randint(0, 100000, 100000).tolist()

    l1=l.copy()
    %timeit quicksort(l1)

    l2=l.copy()
    %timeit _=mergesort(l2)
