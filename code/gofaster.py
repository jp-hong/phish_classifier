

from multiprocessing import Pool
import numpy as np
import pandas as pd


class GoFaster():

    def __init__(self, n_jobs, n_partitions):
        self.n_jobs = n_jobs
        self.n_partitions = n_partitions


    def parallelize(self, df, func):
        df_split = np.array_split(df, self.n_partitions)
        p = Pool(self.n_jobs)
        df = pd.concat(p.map(func, df_split))
        p.close()
        p.join()
        return df

