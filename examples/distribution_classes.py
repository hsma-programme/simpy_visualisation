
'''
Distribution classes

To help with controlling sampling `numpy` distributions are packaged up into 
classes that allow easy control of random numbers.

**Distributions included:**
* Exponential
* Log Normal
* Bernoulli
* Normal
* Uniform
'''

import numpy as np
import math

class Exponential:
    '''
    Convenience class for the exponential distribution.
    packages up distribution parameters, seed and random generator.
    '''
    def __init__(self, mean, random_seed=None):
        '''
        Constructor
        
        Params:
        ------
        mean: float
            The mean of the exponential distribution
        
        random_seed: int, optional (default=None)
            A random seed to reproduce samples.  If set to none then a unique
            sample is created.
        '''
        self.rng = np.random.default_rng(seed=random_seed)
        self.mean = mean
        
    def sample(self, size=None):
        '''
        Generate a sample from the exponential distribution
        
        Params:
        -------
        size: int, optional (default=None)
            the number of samples to return.  If size=None then a single
            sample is returned.
        '''
        return self.rng.exponential(self.mean, size=size)

    
class Bernoulli:
    '''
    Convenience class for the Bernoulli distribution.
    packages up distribution parameters, seed and random generator.
    '''
    def __init__(self, p, random_seed=None):
        '''
        Constructor
        
        Params:
        ------
        p: float
            probability of drawing a 1
        
        random_seed: int, optional (default=None)
            A random seed to reproduce samples.  If set to none then a unique
            sample is created.
        '''
        self.rng = np.random.default_rng(seed=random_seed)
        self.p = p
        
    def sample(self, size=None):
        '''
        Generate a sample from the exponential distribution
        
        Params:
        -------
        size: int, optional (default=None)
            the number of samples to return.  If size=None then a single
            sample is returned.
        '''
        return self.rng.binomial(n=1, p=self.p, size=size)

class Lognormal:
    """
    Encapsulates a lognormal distirbution
    """
    def __init__(self, mean, stdev, random_seed=None):
        """
        Params:
        -------
        mean: float
            mean of the lognormal distribution
            
        stdev: float
            standard dev of the lognormal distribution
            
        random_seed: int, optional (default=None)
            Random seed to control sampling
        """
        self.rng = np.random.default_rng(seed=random_seed)
        mu, sigma = self.normal_moments_from_lognormal(mean, stdev**2)
        self.mu = mu
        self.sigma = sigma
        
    def normal_moments_from_lognormal(self, m, v):
        '''
        Returns mu and sigma of normal distribution
        underlying a lognormal with mean m and variance v
        source: https://blogs.sas.com/content/iml/2014/06/04/simulate-lognormal
        -data-with-specified-mean-and-variance.html

        Params:
        -------
        m: float
            mean of lognormal distribution
        v: float
            variance of lognormal distribution
                
        Returns:
        -------
        (float, float)
        '''
        phi = math.sqrt(v + m**2)
        mu = math.log(m**2/phi)
        sigma = math.sqrt(math.log(phi**2/m**2))
        return mu, sigma
        
    def sample(self):
        """
        Sample from the normal distribution
        """
        return self.rng.lognormal(self.mu, self.sigma)


class Normal:
    '''
    Convenience class for the normal distribution.
    packages up distribution parameters, seed and random generator.
    '''
    def __init__(self, mean, sigma, random_seed=None):
        '''
        Constructor
        
        Params:
        ------
        mean: float
            The mean of the normal distribution
            
        sigma: float
            The stdev of the normal distribution
        
        random_seed: int, optional (default=None)
            A random seed to reproduce samples.  If set to none then a unique
            sample is created.
        '''
        self.rng = np.random.default_rng(seed=random_seed)
        self.mean = mean
        self.sigma = sigma
        
    def sample(self, size=None):
        '''
        Generate a sample from the normal distribution
        
        Params:
        -------
        size: int, optional (default=None)
            the number of samples to return.  If size=None then a single
            sample is returned.
        '''
        return self.rng.normal(self.mean, self.sigma, size=size)

    
class Uniform():
    '''
    Convenience class for the Uniform distribution.
    packages up distribution parameters, seed and random generator.
    '''
    def __init__(self, low, high, random_seed=None):
        '''
        Constructor
        
        Params:
        ------
        low: float
            lower range of the uniform
            
        high: float
            upper range of the uniform
        
        random_seed: int, optional (default=None)
            A random seed to reproduce samples.  If set to none then a unique
            sample is created.
        '''
        self.rand = np.random.default_rng(seed=random_seed)
        self.low = low
        self.high = high
        
    def sample(self, size=None):
        '''
        Generate a sample from the uniform distribution
        
        Params:
        -------
        size: int, optional (default=None)
            the number of samples to return.  If size=None then a single
            sample is returned.
        '''
        return self.rand.uniform(low=self.low, high=self.high, size=size)
