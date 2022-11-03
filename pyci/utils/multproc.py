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
from zipfile import ZipFile

def pool_jobs(func, arglist, ncore, initializer, initargs):
    with Pool(processes=ncore, initializer=initializer, 
              initargs=initargs) as pool:
        async_object = pool.map_async(func, arglist)
        data = async_object.get()
        pool.close()
        pool.join()
    return data


def load_from_npz(npzfilename, array_name):
    zf = ZipFile(npzfilename)
    # figure out offset of .npy in .npz
    info = zf.NameToInfo[array_name + '.npy']
    assert info.compress_type == 0
    zf.fp.seek(info.header_offset + len(info.FileHeader()) + 20)
    # read .npy header
    version = np.lib.format.read_magic(zf.fp)
    np.lib.format._check_version(version)
    shape, fortran_order, dtype = np.lib.format._read_array_header(zf.fp, version)
    offset = zf.fp.tell()
    # create memmap
    return np.memmap(zf.filename, dtype=dtype, shape=shape,
                     order='F' if fortran_order else 'C', mode='rb',
                     offset=offset)
