import random
import numpy as np

class job:
  def __init__(self, id, machines, times = None, n_jobs = None, ML_method = None):
    """
    Here we have our class job which will give us information abour our jobs. We will input the following variables:
    :param id: To identify our job
    :param machines: Just to check how many machines do we have
    :param job_class: It will be an input in our black box
    :param times: If it is None, we will generate a random instance. If it's different to None, we will load our instance.


    Our random instance is created in the firsts lines of the code, getting random integer numbers between 1 and 20 as
    a processing time for each machine.
    """
    self.id = id

    #To create the random instance
    if times is None:
      time_machines = []
      for _ in range(len(machines)):
        time_machines.append(random.randint(1, 20))
    else:
      time_machines = times
    self.time_machines = time_machines

    #To check how many time it takes to compleate all the task in our machine
    self.total_time = sum(time_machines)

    #To use it in our black box

  def black_box(self, h, machine, n):
    """
    Our black box depends on:
      * The fatigate of our machine at the moment of process the job
      * The class of the job (metal, clothes, ...)
      * A random number that we will not know about
    :param h: The weight that our black box has. If 0 there is no black box
    :param machine: The machine which we need to process our job
    :return: We will return a random number of a gamma distribution with the shape that we describe below
    """
    p_ij = self.time_machines[machine.id] #Processing time of the job
    processing_time_bb = p_ij

    #(a) 0.05 * p'_ij if i is multiple of 3 (job i is of class X) AND j has processed between 50% and 75% (both included)
    # of the jobs already
    if machine.processed_jobs >= n*0.5 and machine.processed_jobs <= n*0.75 and (self.job_class == 2 or self.job_class == 3):
      processing_time_bb += h * 0.05 * p_ij

    #(b)0.10 * p'_ij if i is multiple of 3 AND j has processed more than 75% of the jobs already
    elif machine.processed_jobs > n*0.75 and (self.job_class == 2 or self.job_class == 3):
      processing_time_bb += h * 0.1 * p_ij

    #(c)(unknown factor) the quantities above will be increased in an additional 0.02 * p'_ij whenever job i is multiple
    # of 9 (job i is of class X-special)
    if self.job_class == 3:
      processing_time_bb += h * 0.02 * p_ij

    return processing_time_bb
