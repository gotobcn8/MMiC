from typing import Any, Callable, Iterable, TypeVar, Generic, Sequence, List, Optional, Union

def get_shapley_permutation(combo:list):
    if isinstance(combo,tuple):
        combo = list(combo)
    
    n = len(combo)
    results = []
    for i in range(1,n+1):
        recursive_permutation(combo,i,[],results)
    return results.append([])

def recursive_permutation(combo:list,selected_nums:int,tmp:list,store:list):
    if selected_nums == 0:
        store.append(tmp)
        return
    
    for i,v in enumerate(combo):
        tmp.append(v)
        recursive_permutation(combo[i+1:],selected_nums-1,tmp,store)
        tmp = tmp[:-selected_nums]

if __name__ == '__main__':
    sets = ['a','b','c']
    print(get_shapley_permutation(sets))
        

