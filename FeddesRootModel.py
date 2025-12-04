# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 17:40:54 2025

@author: Saurabh
"""

import numpy as np

def f2(psi, psi_a, psi_d, psi_w):
    """
    # plant stress function
    Computes f2(psi) based on a piecewise function.

    Parameters:
    psi   : input pressure head (can be scalar or numpy array)
    psi_a : air entry pressure head
    psi_d : critical pressure head (dry threshold)
    psi_w : wilting point pressure head

    Returns:
    f2_val : result of f2(psi)
    """
    psi = np.array(psi, dtype=float)
    f2_val = np.zeros_like(psi)

    # Case: psi_a > psi > psi_d --> f2 = 1
    mask1 = (psi < psi_a) & (psi > psi_d)
    f2_val[mask1] = 1

    # Case: psi_d >= psi >= psi_w --> f2 = 1 - (psi - psi_d)/(psi_w - psi_d)
    mask2 = (psi <= psi_d) & (psi >= psi_w)
    f2_val[mask2] = 1 - (psi[mask2] - psi_d) / (psi_w - psi_d)

    # psi >= psi_a and psi < psi_w are already 0 by initialization

    return f2_val



def f1(depth,a,Lr,zmax):
    """
    exponential root distribution function
    
    """
    depth = np.array(depth, dtype=float)
    f1 = np.zeros_like(depth)
    mask = depth > Lr
    f1[~mask] = a/Lr *( np.exp(-a)-np.exp(-a*depth[~mask]/Lr) ) / ( (1+a)*np.exp(-a) -1) 
                                                                   
    f1[mask] = 0
    
    
    return f1




def RootUptakeModel(h, RWUData,profileData,Ep_current ):
    zmax, depth = profileData.zmax, profileData.depth

    
    psi_a,psi_d,psi_w, Lr = RWUData.psi_a,RWUData.psi_d,RWUData.psi_w, RWUData.Lr
    a = zmax - Lr # m # measuremnet of how fast root zone declines with depth
    
    f1_vals = f1(depth,a,Lr,zmax) # 1st term of the root water uptake model by Feddes
    f2_vals = f2(h, psi_a, psi_d, psi_w)
    RWU = f1_vals*f2_vals*Ep_current
    
    return RWU















