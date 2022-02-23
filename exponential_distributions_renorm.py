import numpy as np 


def exp1_dist(t, tau, Tmin=0, Tmax=np.inf):
    
    q1 = np.exp(-Tmin/tau) - np.exp(-Tmax/tau)
    return 1/tau*np.exp(-t/tau)/q1

def exp2_dist(t, p1, p2, tau1, tau2, Tmin=0, Tmax=np.inf):
    
    q1 = np.exp(-Tmin/tau1) - np.exp(-Tmax/tau1)
    q2 = np.exp(-Tmin/tau2) - np.exp(-Tmax/tau2)
    
    return p1/tau1*np.exp(-t/tau1)/q1 + p2/tau2*np.exp(-t/tau2)/q2

def exp3_dist(t, p1, p2, p3, tau1, tau2, tau3, Tmin=0, Tmax=np.inf):

    if p3 < 0:
        return np.zeros(t.size)
    
    # Normalization factor
    q1 = np.exp(-Tmin/tau1) - np.exp(-Tmax/tau1)
    q2 = np.exp(-Tmin/tau2) - np.exp(-Tmax/tau2)
    q3 = np.exp(-Tmin/tau3) - np.exp(-Tmax/tau3)
    
    value = p1/tau1*np.exp(-t/tau1)/q1 + p2/tau2*np.exp(-t/tau2)/q2 + p3/tau3*np.exp(-t/tau3)/q3
    if value[value < 0].any():

        raise ValueError('negative probability!')
    return value

def exp4_dist(t, p1, p2, p3, p4, tau1, tau2, tau3, tau4, Tmin=0, Tmax=np.inf):
    
    assert p4 == 1 - p1 - p2 - p3
    
    if p4 < 0:
        return np.zeros(t.size)
    
    # Normalization factor
    q1 = np.exp(-Tmin/tau1) - np.exp(-Tmax/tau1)
    q2 = np.exp(-Tmin/tau2) - np.exp(-Tmax/tau2)
    q3 = np.exp(-Tmin/tau3) - np.exp(-Tmax/tau3)
    q4 = np.exp(-Tmin/tau4) - np.exp(-Tmax/tau4)
    
    value = p1/tau1*np.exp(-t/tau1)/q1 + p2/tau2*np.exp(-t/tau2)/q2 + p3/tau3*np.exp(-t/tau3)/q3 + p4/tau4*np.exp(-t/tau4)/q4
    
    if value[value < 0].any():

        raise ValueError('negative probability!')
    return value


def log_bin(data, bin_width=0.2, treat_zero_bins=True):
    '''
    Compute log bins for the input data
    '''
    bsize = bin_width
    bin_edges = 10**(np.arange(np.log10(min(data)), np.log10(max(data)) + bsize, bsize))
    values, bins = np.histogram(data, bins=bin_edges, density=True)
    if treat_zero_bins:
        # combine bins until they contain at least one data point (for y-log plots)
        izeros = np.where(values == 0)[0]
        j = 0
        while j < len(izeros):
            i = j
            j += 1
            while j < len(izeros) and (izeros[j] - izeros[j-1]) == 1:
                j += 1
            values[izeros[i]: (izeros[i] + j - i + 1)] = np.sum(values[izeros[i]:(izeros[i]+j-i+1)])/(j-i+1)
    # geometric average of bin edges    
    centers = (bins[1:] * bins[:-1])**0.5
    
    return centers, values

def LogLike(parameters, model_func, data):
    
    if type(parameters) != list:
        parvals = list(parameters.valuesdict().values())
    else:
        parvals = parameters
        
    Pi = model_func(data, *parvals)
    # if np.any(np.where(Pi == 0)):
        # print('0 probabilities encountered')

    Pi[np.where(Pi == 0)] = 1e-30

    
    LLike = -np.sum(np.log(Pi))

    return LLike