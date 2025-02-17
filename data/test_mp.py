from random import random
from time import sleep
from multiprocessing.pool import Pool
from multiprocessing import set_start_method
import numpy as np
 
# task executed in a worker process
def task(identifier):
    # generate a value
    value = random()
    # report a message
    print(f'Task {identifier} executing with {value}', flush=True)
    # block for a moment
    sleep(1)
    # return the generated value
    return value
 
data = list(np.arange(0,40))
# One method of spawning the child processes is using set_start_method('fork'), it only works where 'fork' is supported (Unix/MacOS)
set_start_method('fork')
with Pool(4) as pool:
    # issue tasks to the process pool
    pool.imap(task, data, chunksize=10)
    # shutdown the process pool
    pool.close()
    # wait for all issued task to complete
    pool.join()

# Another method is to protect the entry point using if __name__ == "__main__"
'''
if __name__ == "__main__":
    with Pool(4) as pool:
        # issue tasks to the process pool
        pool.imap(task, range(40), chunksize=10)
        # shutdown the process pool
        pool.close()
        # wait for all issued task to complete
        pool.join()
'''

