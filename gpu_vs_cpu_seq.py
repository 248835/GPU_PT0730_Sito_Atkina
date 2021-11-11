# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from typing import List

import math
from timeit import default_timer as timer
import numpy as np
import cupy as cp
import sympy
# import os
# os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
# os.environ["NUMBA_CUDA_DEBUGINFO"] = "1"
from numba import cuda


@cuda.jit
def cross_out_multiples(number, n, out):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, out.shape[0], stride):
        p = i * number
        if number < p < n:
            out[p] = 0


@cuda.jit
def create_result_array(a_device, c_device):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    if start == 0:
        return

    for i in range(start, a_device.shape[0], stride):
        if a_device[i - 1] != a_device[i]:
            c_device[a_device[i]] = i



def sieveGpu(n: int) -> np.ndarray:
    A = np.ones(n, dtype=np.int32)
    A[0] = 0
    a_device = cp.asarray(A, dtype=np.int32) #cuda.to_device(A)
    for i in range(2, math.floor(math.sqrt(n)) + 1):
        if a_device[i] == 1:
            cross_out_multiples[64, 64](i, n, a_device)

    c_device = cuda.device_array(len(A), dtype=np.int32)
    #print(a_device[:15])
    cp.cumsum(a_device, dtype=np.int32, out=a_device)
    count = cp.take(a_device, [-1])
    count = count.tolist()[0] + 1
    #print(a_device[:15])
    # A = a_device.copy_to_host()
    # print(A[:15])
    # A = A.cumsum(dtype=np.int32)
    # print(A[:15])
    # a_device = cuda.to_device(A)
    create_result_array[64, 64](a_device, c_device)
    C = c_device.copy_to_host()
    #print(C[:15])

    # for i in range(2, len(A)):
    #     if A[i] == 1:
    #         C.append(i)

    return C[2:count]


def sieveCpu(n: int) -> np.ndarray:
    A = np.ones(n)
    A[0] = 0
    for i in range(2, math.floor(math.sqrt(n)) + 1):
        if A[i] == 1:
            p = i * i
            while p < n:
                A[p] = 0
                p += i

    C = []
    for i in range(2, len(A)):
        if A[i] == 1:
            C.append(i)

    return np.array(C)


def measure(n: int):
    print("For n =", n)

    start = timer()
    cpu_result = sieveCpu(n)
    end = timer()
    print("\tCPU: ", end - start)

    start = timer()
    gpu_result = sieveGpu(n)
    end = timer()
    print("\tGPU: ", end - start)


def testValidity(n):
    cpu_primes = sieveCpu(n)
    gpu_primes = sieveGpu(n)
    cpuGpuEqual = np.array_equal(cpu_primes, gpu_primes)
    external_lib_primes = np.array(list(sympy.sieve.primerange(n)))
    cpuExtEqual = np.array_equal(cpu_primes, external_lib_primes)
    primesValid = cpuGpuEqual and cpuExtEqual
    print("Algorithms results are valid:", primesValid)
    if primesValid:
        print(cpu_primes[:15])
    else:
        print(cpu_primes[:15])
        print(gpu_primes[:15])
        print(external_lib_primes[:15])


if __name__ == '__main__':
    #precompile
    sieveGpu(10)

    testValidity(1000000)
    #testValidity(100000000)

    measure(1000000)
    measure(10000000)
    measure(100000000)
