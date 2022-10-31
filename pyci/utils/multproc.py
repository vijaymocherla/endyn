#!/usr/bin/env python
#
#   Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
""" multproc.py
Contains parallelisation routines writtein for pyci using
multiprocessing library natively available in python.
"""
from multiprocessing import Pool, Queue, RawArray
from functools import partial
import numpy as np

def pool_jobs(func, arglist, ncore, initializer, initargs):
    with Pool(processes=ncore, initializer=initializer, 
              initargs=initargs) as pool:
        async_object = pool.map_async(func, arglist)
        data = async_object.get()
        pool.close()
        pool.join()
    return data

