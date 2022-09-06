
class machine:

  def __init__(self, id):
    """
    To describe our machine.

    We will take care about the time and the number of the machines
    """
    self.id = id
    self.processed_jobs = 0
    self.time = 0
    
    # Don't we have to define a different time per each job type?
    # Maybe a 'matrix of times' would be better
    
    # Juanfran: Why do you think we need a matrix? What I thought is to add in "time" 
    # the processing time that the machine spends when some jobs are processing in each machine.
    # Do you want to have a matrix to know what is the processing time for each job?
    # Our processing times for each job is in "job.time_machines".
    
    # Got it. Right, I thought machine.time was our job.time_machines
