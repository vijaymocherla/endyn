#!/usr/bin/env python
#
#   Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
""" multproc.py
Contains parallelisation routines writtein for pyci using
multiprocessing library natively available in python.
"""
from multiprocessing import Pool

# use better func name
def pool_jobs(pfunc, arglist, ncore):
    with Pool(processes=ncore) as pool:
        async_object = pool.map_async(pfunc, arglist)
        data = async_object.get()
        pool.close()
        pool.join()
    return data

