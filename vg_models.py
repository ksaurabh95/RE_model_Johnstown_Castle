# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 12:32:39 2025

@author: Saurabh
"""

# functions files to estmate the K, theta, C, Ss based on van genuchten parameters
# theta = soil moisture 
# K = soil hydrualic conductivity 
# C = specific moisture capacity  
# Ss = specific storage
 
import numpy as np


def VG_theta( h , vgData ):
    """
    calculation of soil moisture based on  Van Genuchten relationship (1981)
    thetar = residual moisture content, thetas = saturated soil moisture , h = presure head, 
    hs = air entry presure head , N = Van Genuchten paramter , Ss =  specific storage
    """
    # Ksat = np.array(vgData["Ksat (m/day)"])  # m/day
    thetas = np.array(vgData["thetas"])
    thetar = np.array(vgData["thetar"])
    alpha = np.array(vgData["alpha (m-1)"])
    N = np.array(vgData["N"])
    m = 1- 1/N    
    beta = pow( abs(alpha*h), N )
    y = thetar + (thetas - thetar)* pow( 1+ beta, -m )
    
    # when h > 0, theta = thetas
    tempI = h > 0 
    y[tempI] = thetas[tempI]
    
    return y

def VG_K( h ,vgData):
    """
    calculation of soil hydraulic conductivity  based on modified Van Genuchten and  Nelson (1985)
    thetar = residual moisture content, thetas = saturated soil moisture , h = presure head, 
    hs = air entry presure head , N = Van Genuchten paramter , Ss =  specific storage
    n_eta =  Relativepermeabilityexponen
    """
    Ksat = np.array(vgData["Ksat (m/day)"])  # m/day
    alpha = np.array(vgData["alpha (m-1)"])
    N = np.array(vgData["N"])
    m = 1- 1/N    
    beta = pow( abs(alpha*h), N )
    m = 1 - 1/N
    y = Ksat*( ( (1+ beta)**(-5*m/2) ) * (  ( (1+ beta)**m  - beta**m  )**2 ) )

    tempI = h >= 0 
    y[tempI] = Ksat[tempI]
        
    return y

def Van_Genuchten_specific_moisure_capacity( h ,vgData ):
    """
    calculation of soil specific moisture capacity based on 
    modified Van Genuchten and Nelson (1985)
    thetar = residual moisture content, thetas = saturated soil moisture , h = presure head, 
    hs = air entry presure head , N = Van Genuchten paramter , % Ss =  specific storage
    """
    thetas = np.array(vgData["thetas"])
    thetar = np.array(vgData["thetar"])
    alpha = np.array(vgData["alpha (m-1)"])
    N = np.array(vgData["N"])
    n_eta = 0.5*np.ones(len(thetas))
    m = 1- 1/N
    hs = 1./alpha
    beta = pow( abs(alpha*h), N )
    y = (N - 1)*(thetas - thetar)*pow(abs(h),N-1 )/( pow(abs(hs),N)*pow(1 + beta, m +1) )
    
    tempI = h > 0 
         
    y[tempI] = 0

        
    return y

def Van_Genuchten_Ss( h ,vgData ):
    """
    calculation of soil specific storage Ss based on 
    modified Van Genuchten and Nelson (1985)
    thetar = residual moisture content, thetas = saturated soil moisture , h = presure head, 
    hs = air entry presure head , N = Van Genuchten paramter , % Ss =  specific storage
    """
    thetas = np.array(vgData["thetas"])
    thetar = np.array(vgData["thetar"])
    alpha = np.array(vgData["alpha (m-1)"])
    N = np.array(vgData["N"])
    # n_eta = 0.5*np.ones(len(thetas))
    m = 1- 1/N
    hs = 1./alpha
    beta = pow( abs(alpha*h), N )
    # it is evaluated at h = 0 
    y = 0* np.empty(len(thetas))
    # y = (N - 1)*(thetas - thetar)*pow(abs(h),N-1 )/( pow(abs(hs),N)*pow(1 + beta, m +1) ) # to be used for modified vg and nelson formula
    


        
    return y



def vgModel(h ,vgData):
    
    theta = VG_theta( h , vgData )
    K = VG_K( h , vgData )
    C = Van_Genuchten_specific_moisure_capacity( h ,vgData )
    Ss = Van_Genuchten_Ss( h ,vgData )


    return theta, K,  C, Ss



# example 
# initial condition 
# h_i_obs_hpa = np.array([60,45,1,-13,-49,-70 , -70])
# h_i_obs  = - 1.0197*h_i_obs_hpa/100 # converting in m 
# depth_obs = [0.15,0.3,0.45,0.60,0.90,1.20, 3.0]
# h_i = np.interp(depth, depth_obs, h_i_obs)
# theta_i,K_i, C_i, Ss_i  = vgModel( h_i , vgData )















def Van_Genuchten_moisure( h , thetar, thetas , alpha, N):
    """
    calculation of soil moisture based on  Van Genuchten relationship (1981)
    thetar = residual moisture content, thetas = saturated soil moisture , h = presure head, 
    hs = air entry presure head , N = Van Genuchten paramter , Ss =  specific storage
    """
    m = 1- 1/N
    y =  np.empty([len(h)])
    tempI = h < 0 
    if h <= 0 :
        beta = pow( abs(alpha*h), N )
        y = thetar + (thetas - thetar)* pow( 1+ beta, -m )  
    else: 
        y = thetas
    
    return y


# def Van_Genuchten_K( h , Ksat, thetar, thetas , alpha, N, n_eta):
#     """
#     calculation of soil moisture based on modified Van Genuchten and  Nelson (1985)
#     thetar = residual moisture content, thetas = saturated soil moisture , h = presure head, 
#     hs = air entry presure head , N = Van Genuchten paramter , Ss =  specific storage
#     n_eta =  Relativepermeabilityexponen
#     """
#     # y = np.empty([len(h)])
#     m = 1 - 1/N
#     beta = pow( abs(alpha*h), N )
#     Se = pow( 1+ beta,-m)
#     # y =  Ksat*pow(Se,n_eta)*pow( ( 1 - pow((1 - pow(Se,1/m)),m) ) , 2) 
#     if h < 0:    
#        y =  Ksat*pow(Se,n_eta)*pow( ( 1 - pow((1 - pow(Se,1/m)),m) ) , 2) 
#     else: y = Ksat 
    
#     return y



# def Van_Genuchten_specific_moisure_capacity( h , thetar, thetas, alpha, N ):
#     """
#     calculation of soil specific moisture capacity based on 
#     modified Van Genuchten and Nelson (1985)
#     thetar = residual moisture content, thetas = saturated soil moisture , h = presure head, 
#     hs = air entry presure head , N = Van Genuchten paramter , % Ss =  specific storage
#     """
#     # tempI = h > 0
#     m = 1 - 1/N 
#     hs = 1./alpha
#     if h <= 0 :
#         beta = pow( abs(alpha*h), N )
#         y = (N - 1)*(thetas - thetar)*pow(abs(h),N-1 )/( pow(abs(hs),N)*pow(1 + beta, m +1) ) 
#     else: 
#         y = 0
        
#     return y
