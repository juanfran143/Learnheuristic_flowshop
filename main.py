from job import job
from sol import sol
from machine import machine
from utils import *
import pickle
from ML import train_random_forest
from job import *
from machine import *
from sol import *

if __name__ == '__main__':
    '''
    We generate a random instance to run our algorithm (Go to the class job to see this random generation).
    
    After that, we select the time (max_time), the 'h' of our black box (explained in the class 'job'), our beta 
    (to do our biased randomize), our jobs and machines. After that, we run our algorithm (multistart_NEH) 
    '''

    read_data('test.txt')
    """ 
    jobs = [job(0, 3, times = [5,6,4]), job(1, 3, times = [2,7,5]), job(2, 3, times = [5,3,4]), job(3, 3, times = [4, 4, 3])]
    machines = [machine(1), machine(2), machine(3)]
    solution1 = sol(jobs, machines, beta=0.2)
    solution2 = sol(jobs, machines)

    for _ in range(10):
        solution1 = sol(jobs, machines)
        print("Aceleration:")
        solution1.NEH_acceleration()
        #print(solution1.makespan)
        print("Deterministic:")
        print(solution1.deterministic_makespan())
    #solution1.NEH()
    """