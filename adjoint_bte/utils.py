import h5py
import mpmath
import numpy as np
import scipy.constants as const
import torch
from pymatgen.core.periodic_table import Element

def compute_heatcap(freq_rad, dos, T):
    
    '''

    Parameters
    ----------
    freq_rad : numpy array
        Frequency in rad/s.
    dos : numpy array
        DESCRIPTION.
    T : numpy array
        Temperature.

    Returns
    -------
    Mode heat capacity.
    f_BE = 1/(exp(hbar*omega/(k*T))-1)
    dfdT = omega*hbar*csch(omega*hbar/(2*k*T))**2/(4*k*T**2)

    '''
    
    hbar = const.hbar / const.value('electron volt') 
    k = const.k / const.value('electron volt')

    # y = np.array([float(mpmath.csch(hbar * x / (2 * k * T.cpu().numpy()))) ** 2 for x in freq_rad.cpu().numpy()])
    y = 1 / torch.sinh(hbar * freq_rad / (2 * k * T)) ** 2
    # dfdT = - (1/4)*(1/(k*T))*y
    dfdT = hbar * freq_rad * y / (4 * k * T ** 2)
    C = hbar * torch.einsum('kl,k->kl', dos, freq_rad * dfdT) # freq_rad * dos * dfdT
    
    return C


def compute_heatcap_new(freq_rad, dos, T, struct):
    
    '''

    Parameters
    ----------
    freq_rad : numpy array
        Frequency in rad/s.
    dos : numpy array
        DESCRIPTION.
    T : numpy array
        Temperature.

    Returns
    -------
    Mode heat capacity.
    f_BE = 1/(exp(hbar*omega/(k*T))-1)
    dfdT = omega*hbar*csch(omega*hbar/(2*k*T))**2/(4*k*T**2)

    '''
    
    hbar = const.hbar
    k = const.k
    dos_normed = 3 * struct.num_sites * dos / torch.trapz(dos, freq_rad, dim=0)

    # y = np.array([float(mpmath.csch(hbar * x / (2 * k * T.cpu().numpy()))) ** 2 for x in freq_rad.cpu().numpy()])
    y = 1 / torch.sinh(hbar * freq_rad / (2 * k * T)) ** 2
    # dfdT = - (1/4)*(1/(k*T))*y
    dfdT = hbar * freq_rad * y / (4 * k * T ** 2)
    Cv = hbar * torch.einsum('kl,k->kl', dos_normed, freq_rad * dfdT) # freq_rad * dos * dfdT
    Cv_per_cubicmeter = 1e30 / struct.volume * Cv # [J/K/m3]

    return Cv_per_cubicmeter

def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step * (1 - np.exp(-t * rate / step)))

def compute_interface_thermal_conductance(freq, dos, T, struct, vg, T12):
    
    Cv = compute_heatcap_new(freq, dos, T, struct)

    return None