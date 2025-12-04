# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 12:26:45 2025

@author: Saurabh
"""

import numpy as np
from scipy.optimize import fsolve
import pandas as pd
from scipy.interpolate import interp1d
from vg_models import vgModel
from FeddesRootModel import RootUptakeModel
# from dataclasses import dataclass, field
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix
from vg_models import Van_Genuchten_moisure, Van_Genuchten_K


def assign_vg(depth, df, extend_to):
    # extend last horizon if needed
    if extend_to and extend_to > df["Depth_max"].iloc[-1]:
        last = df.iloc[-1].copy()
        last["Depth_min"] = last["Depth_max"]
        last["Depth_max"]   = extend_to
        df = pd.concat([df, pd.DataFrame([last])], ignore_index=True)

    # assign parameters to each depth
    out = []
    for i in depth:
        row = df[(i >= df["Depth_min"]) & (i < df["Depth_max"])].iloc[0]
        out.append({"depth": i, **row.to_dict()})
        
    out = pd.DataFrame(out)
    out = out.drop(
            ['Horizon', 'Depth_min', 'Depth_max'],
            axis=1)   
    out['z'] = extend_to - depth
    return out


def soil_grid(zmax,dz):
    # code to generate soil grid, provide midpoints for all grid locations
    # z is measured upwards, zmin is where is the lower boundary condition or bottom taken as 0, zmax is at top.  depth is measured downwardsfrom surface    
    zmin = 0
    z = np.arange(zmin + dz/2 ,zmax - dz/2,dz )
    dz_all = dz * np.ones(len(z))        
    depth = zmax - z 
            
    return z,dz_all,depth


def central_zone(A, RHS, h_current, theta_current, K_current, h_old, theta_old, C, Ss, RWU,profileData,dt):
    
    z, dz_all = profileData.z,profileData.dz_all    
    znodes = len(dz_all)
    
    # theta_current,K_current, C, Ss  = vgModel( h_current , vgData )
    # theta_old,_, _, _  = vgModel( h_old , vgData )

       
    for i in range(1,znodes-1):
                
        d1z = dz_all[i-1]
        d2z = dz_all[i]
        d3z = dz_all[i+1]
        
        h2  = h_current[i]       
        h2_old = h_old[i]
       
        K1Z = K_current[i-1]
        K2  = K_current[i]
        K3Z = K_current[i+1]
    
        # calculation of s
        s =(theta_current[i]-theta_old[i])/dt          
      # calculation of p1 
        p1 = C[i]/dt
        # calculation of p2
        p2 = 0; # Sw*Ss/dt; since Sw=0
            
        Kszminus = ( K2*d2z + K1Z*d1z ) / ( d1z+d2z )
        CNZminusmean = -Kszminus/( (d2z+d1z)/2)
        c = -CNZminusmean/d2z;             
      
        # calculation of g
        
        Kszplus = ( K2*d2z + K3Z*d3z )/ (d3z+d2z)
        CNZplusmean = -Kszplus/( (d2z+d3z)/2)
        g = -CNZplusmean/d2z  
        # calculation of d          
        d = -(  c + g + p1 + p2)
        # forming the A matrix and RHS vector
        A[i,i-1] = c   # coeffiecient of H(k-1)
        A[i,i] = d      # coeffiecient of H(k)
        A[i,i+1] = g  # coeffiecient of H(k+1)
        RHS[i] = s - p1 *( h2 + z[i] ) - p2*( h2_old + z[i])  + RWU[i] # RHS at i
    
    return A,RHS



def bottom_free_drainage(A, RHS,h_current, theta_current, K_current, h_old, theta_old, C, Ss, RWU,profileData,dt):
    
    z, dz_all = profileData.z,profileData.dz_all    
    znodes = len(dz_all)
    
    i = 0   # free drainage at the bottom 
 
    d2z = dz_all[i]
    d3z = dz_all[i+1]
    
    h2  = h_current[i]
    
    h2_old = h_old[i]

    K2  = K_current[i]
    K3Z = K_current[i+1]
   

    # calculation of s
    s =(theta_current[i]-theta_old[i])/dt        
  # calculation of p1 
    p1 = C[i]/dt
    # calculation of p2
    p2 = 0; 

    Kszplus = ( K2*d2z + K3Z*d3z )/ (d3z+d2z)
    CNZplusmean = -Kszplus/( (d2z+d3z)/2)
    g = -CNZplusmean/d2z 
    d = -(  g + p1 + p2)
    A[i,i] = d      # coeffiecient of H(k)
    A[i,i+1] = g  # coeffiecient of H(k+1)
    RHS[i] = s - p1 *( h2 + z[i] ) - p2*( h2_old + z[i]) + K2/d2z + RWU[i]  # RHS at i
  
    return A,RHS


def top_flux_boundary(A, RHS,h_current, theta_current, K_current, h_old, theta_old, C, Ss, RWU,profileData,dt ,q_current):
    
    z, dz_all = profileData.z,profileData.dz_all    
    znodes = len(dz_all)
    
    i = znodes-1
    d1z = dz_all[i-1]
    d2z = dz_all[i]
    h2  = h_current[i]
    
    h2_old = h_old[i]

    
    K1Z = K_current[i-1]
    K2  = K_current[i]
        
    # calculation of s
    s =(theta_current[i]-theta_old[i])/dt        
  # calculation of p1 
    p1 = C[i]/dt
    # calculation of p2
    p2 = 0;         
    
    Kszminus = ( K2*d2z + K1Z*d1z ) / ( d1z+d2z )
    CNZminusmean = -Kszminus/( (d2z+d1z)/2)
    c = -CNZminusmean/d2z;             

    d = -(  c +  p1 + p2)
    
    A[i,i-1] = c   # coeffiecient of H(k-1)
    A[i,i] = d      # coeffiecient of H(k)
    RHS[i] = s - p1 *( h2 + z[i] ) - p2*( h2_old + z[i]) - q_current/d2z + RWU[i]

    
    return A,RHS

def fixed_headBC(A, RHS,z,iloc = 0, h_fixed = 0 ):
    # code to perform fixed head boundary condition at both top and bottom boundary condition 
    # by default I am applying at the bottom boundary condition 
    # iloc specify the node location, i = znodes-1 for top and i = 0 for bottom boundary condition 
    A[iloc, :] = 0 # zero entire row
    A[iloc,iloc] = 1     # coeffiecient of H(k)
    RHS[iloc] =  h_fixed + z[iloc]   # RHS at i
  
    return A,RHS




# def ponding_flux(h_curr,K_curr, C , vgData,dz_all,z,Se_top= 0.999):
    
#     # estimation of flux at the top when ponding condition has been reached, i.e., Se >= 0.999 
    
#     thetas = np.array(vgData["thetas"])
#     thetar = np.array(vgData["thetar"])
#     alpha = np.array(vgData["alpha (m-1)"])
#     N = np.array(vgData["N"])
#     n_eta = 0.5*np.ones(len(thetas))
#     Ksat = np.array(vgData["Ksat (m/day)"])  # m/day
    
#     znodes = len(thetas)
    
#     i = znodes-1
    
#     thetas_cur = thetas[i]
#     thetar_cur = thetar[i]
#     alpha_cur = alpha[i]
#     N_cur = N[i]
#     Ksat_cur = Ksat[i]
#     n_eta_cur = n_eta[i]
#     m = 1 - 1/N_cur
    
    
#     h_top = - (1.0 / alpha_cur) * (Se_top**(-1.0/m) - 1.0)**(1.0 / N_cur)
#     # K_top = Ksat_cur  # *(Se_top**n_eta_cur)*( ( 1 - (1 - Se_top**(1/m))**m  )**2)
#     K_top = Ksat_cur*(Se_top**n_eta_cur)*( ( 1 - (1 - Se_top**(1/m))**m  )**2)

#     K2 = K_curr[i]
#     h2 = h_curr[i]
#     d2z = dz_all[i]
    
#     Kszminus = ( K2 + K_top )  / 2 
#     # Kszminus = ( K2 )
#     # Kszminus = ( K_top )


#     q = -Kszminus*( ( h2- h_top)/(d2z/2) -1  )
    
    
#     return q 


def ponding_flux(h,Ksat, thetar, thetas , alpha, N, n_eta,dz_all,z):
    # estimation of flux at the top when ponding condition has been reached, i.e., Se >= 0.999 
    Se_top = 0.999
    
    # code to estimate the pressure head the point 
    znodes = len(dz_all)
    
    i = znodes-1
    thetas_current = thetas[i]
    thetar_current = thetar[i]
    alpha_current = alpha[i]
    N_current = N[i]
    Ksat_current = Ksat[i]
    n_eta_current = n_eta[i]

  
    theta_top = Se_top*(thetas_current - thetar_current) + thetar_current
    m = 1-1/N_current
    
    K_top = Ksat_current*(Se_top**n_eta_current)*( ( 1 - (1 - Se_top**(1/m))**m  )**2)

    f = lambda h: Van_Genuchten_moisure(h, thetar_current, thetas_current, alpha_current, N_current) - theta_top

    # # Initial guess (h must be negative in unsaturated soil)
    # h_top  = fsolve(f, x0=-1)[0]
    h_top = - (1.0 / alpha_current) * (Se_top**(-1.0/m) - 1.0)**(1.0 / N_current)
    
    h2  = h[i]
    beta =   pow( abs(alpha_current*h2), N_current )
    Se_h2 = pow( 1+ beta, -m )      
    K2 = Ksat_current*(Se_h2**n_eta_current)*( ( 1 - (1 - Se_h2**(1/m))**m  )**2)
    
    d2z = dz_all[i]
  
    Kszminus = ( K2 + K_top )  / 2 
    # Kszminus = ( K2 )
    # Kszminus = ( K_top )


    # q = -Kszminus*( ( h2- h_top)/(d2z/2) -1  )
    q = -Kszminus*( ( h2- h_top)/(d2z) -1  )
        
    
    return q



def Richards_Solver(profileData, timeData,vgData,MetData,RWUData, h_ini ):
    
    z, dz_all, depth = profileData.z,profileData.dz_all, profileData.depth
    # extracting the vg parameters 
    Ksat = np.array(vgData["Ksat (m/day)"])  # m/day
    thetas = np.array(vgData["thetas"])
    thetar = np.array(vgData["thetar"])
    alpha = np.array(vgData["alpha (m-1)"])
    N = np.array(vgData["N"])
    n_eta = 0.5*np.ones(len(thetas))
    # the meterological data 
    q = np.array(MetData["rain_mm"]) /1000 # rainfall m/day
    PET = np.array(MetData["pet_mm_per_day"] )/1000   # m/day
    # PET = PET - evap
    # time specs to be used to run the code 
    dt_min = timeData.dt_min # min time step 
    dt_max = timeData.dt_max # max time step 
    plus_factor = timeData.plus_factor  # factor with which time step dt will either increase or decrease
    dec_factor = timeData.dec_factor # decrease factor for time step    
    tmin = timeData.tmin  # starting time in day
    tmax = timeData.tmax  # max time in day
    dt = timeData.dt; #in day  
    time_given = timeData.time_given
    # tnodes = len(rain)

    tnodes = 30*round( 2*tmax/( dt_min + dt_max )  + 0.5*tmax/dt_max )  #  max avialabe timesteps 
    t = np.empty([tnodes]);
    # t = np.NaN
    t[0] = tmin 

    # preallocate h and theta 
    znodes = len(z) # nos of nodes 

    h = np.zeros([znodes,tnodes])
    theta=np.zeros([znodes,tnodes])
    K = np.zeros([znodes,tnodes])
    C = np.zeros([znodes,tnodes])
    Se = np.empty([znodes,tnodes])
    RWU = np.empty([znodes,tnodes])
    Ta =  np.empty([tnodes])
    Ea =  np.empty([tnodes])
    q_ponding_flux = np.empty([tnodes])
    q_rain = np.empty([tnodes])
    # nodal depth till which roots are present
    nodes_depth_Lr = depth <= RWUData.Lr

    # initial condition 
    h_i = np.interp(profileData.depth, h_ini['depth'], h_ini['h'])
    h[:,0] = h_i # depth - z_wt # given 
    # initial moisture, hydrualic conductivity  condition
    theta[:,0], K[:,0],  _, _ = vgModel( h[:,0] , vgData )
    Se[:,0] =( theta[:,0] - vgData["thetar"] ) / (vgData["thetas"] - vgData["thetar"]  )

    # stopping criteria
    threshold = .001
    MaxIterations = 30
    Error = np.zeros([tnodes,MaxIterations]) # storing RMSE error at each time step and iteration 
    residual = np.zeros([tnodes])
    # begin simulation 
    n = 0  # initialize the timestep counter explicitly    

    # Creating the interpolation function with step-wise behavior
    rain_interp_func = interp1d(time_given, q, kind='next', bounds_error=False, fill_value='extrapolate')
    Ep_interp_func = interp1d(time_given, PET, kind='next', bounds_error=False, fill_value='extrapolate')
    
    print('Solver Starting ....')


    while n < tnodes:
        
        Repeat_itr = False  # timestep required to repeated due to max iteration condition 
        current_time = t[n] + dt # unknown time for which prediction will be done
        # # Interpolated rainfall and ET values
        q_current = rain_interp_func(current_time)
        q_rain[n+1]  = q_current    
        Ep_current = Ep_interp_func(current_time)

        h[:,n+1] = h[:,n]
        # print(n)
        Error[n,:] = np.nan
        h_old = h[:,n]
        theta_old =  theta[:,n]
        
        
  # performing iteration within each time step      
        for m in range(MaxIterations):
            RHS = np.empty([znodes])
            A = lil_matrix((znodes,znodes))         # good for assembly
            theta[:,n+1], K[:,n+1],  C[:,n+1], Ss = vgModel( h[:,n+1] , vgData )
            h_curr = h[:,n+1] 
            theta_curr = theta[:,n+1] 
                    
            RWU[:,n+1] = RootUptakeModel(h[:,n+1], RWUData,profileData,Ep_current )             
            Ta[n+1 ] = np.trapezoid(RWU[nodes_depth_Lr,n+1], x=z[nodes_depth_Lr])*1000 
            Ea [n+1 ] = Ep_current - Ta[n+1 ]  
            
            q_i = q_current  
            q_ponding_flux[n+1] = ponding_flux(h[:,n+1],Ksat, thetar, thetas , alpha, N, n_eta,dz_all,z)
            # q_ponding_flux[n+1] = ponding_flux(h_curr,K[:,n+1], C[:,n+1] , vgData,dz_all,z,Se_top= 0.995)  
            
            if q_current > q_ponding_flux[n+1]: 
                q_i = q_ponding_flux[n+1]
                # print('ponding')
                A, RHS = fixed_headBC(A, RHS,z,iloc = znodes-1, h_fixed = -0.001)  # Top boundary condition at saturation

            else: A, RHS = top_flux_boundary(A, RHS,h_curr, theta_curr, K[:,n+1], h_old, theta_old, C[:,n+1], Ss, RWU[:,n+1],profileData,dt ,q_i)
                  
            A, RHS = central_zone(A, RHS, h_curr, theta_curr, K[:,n+1], h_old, theta_old, C[:,n+1], Ss, RWU[:,n+1],profileData,dt)
            # A, RHS = bottom_free_drainage(A, RHS,h_curr, theta_curr, K[:,n+1], h_old, theta_old, C[:,n+1], Ss, RWU[:,n+1],profileData,dt)   # free drainage       
            A, RHS = fixed_headBC(A, RHS,z,iloc = 0, h_fixed = 0.001)  # Bottom boundary condition 
            X_input = h[:,n+1]+z
            A = A.tocsr()   # convert to CSR for solving

            try:
                # X = np.linalg.solve(A, RHS)
                X = spsolve(A, RHS)

                # X = np.linalg.inv(A).dot(RHS)
            except np.linalg.LinAlgError:  # to supress error and simulation break
                Repeat_itr = True
                dt = dt*(1-plus_factor)

                n = n -1# time step will not change 
                break  # Exit the iteration loop to repeat the timestep with smaller dt

            Error[n,m] =  np.sqrt(np.sum(pow(X-X_input,2)))  # RMSE
            residual [n] = Error[n,m]
            
            # updating the presure head, theta, K and C 
            h[:,n+1] = X-z
            theta[:,n+1], K[:,n+1],  C[:,n+1], _ = vgModel( h[:,n+1] , vgData )
            Se[:,n+1] = ( theta[:,n+1] - vgData["thetar"] ) / (vgData["thetas"] - vgData["thetar"]  )
            Ta[n+1] = np.sum(RWU[:,n+1]*dt)

            if Error[n,m] <= threshold:
                break
            
        else: # repeating the loop if only iteration is not repeated
            Repeat_itr = True
            # break 
            n = n -1# time step will not change 
            dt = dt / 3 # current value of dt is reduced by a third
            # print("repeat")
        

        t[n + 1] = t[n] + dt # moving to next time step
        # n = n + 1  # moving to next time node
        time_elasped = t[n + 1]
        # print("simulation run time =",time_elasped)

        if n % 200 == 0:    
            print("simulation run time =",time_elasped)    
        n = n + 1 # moving to the next iteration     
        
        # adaptive time settings [Mls, 1982; Šimunek et al., 1992]:
        if m <= 3 and dt >= dt_min and dt <= dt_max:
            dt = dt * (1 + plus_factor)  # for next time step
        elif m >= 7 and dt >= dt_min and dt <= dt_max:
            dt = dt * (1 - dec_factor)  # for next time step
        # limiting dt range
        dt = np.clip(dt, dt_min, dt_max)
        
        # ensuring time_elasped does not increase t_max
        time_left = tmax - time_elasped
        if time_left < dt:
            dt = time_left
            
        if time_elasped == tmax:
            break



    # removing the extra nodes from the required variables 
    t = t[0:n+1]
    residual = residual[0:n+1]
    q_rain = q_rain[0:n+1]
    q_ponding_flux = q_ponding_flux[0:n+1]
    # abs_error_mass_bal = abs_error_mass_bal[0:n+1] 
    # rel_error_mass_bal = rel_error_mass_bal[0:n+1] 
    h = h[:, 0:n+1] 
    theta = theta[:, 0:n+1] 
    K = K[:,0:n+1]

    Se = Se[:, 0:n+1] 
    RWU = RWU[0:n+1] 
    Error=  Error[0:n+1,:]
    Ea=  Ea[0:n+1]
    Ta=  Ta[0:n+1]


    return theta, K, h, Se,Ea,RWU,t, Ta, Error,q_ponding_flux,q_rain








def spatial_interpolation(profileData,timeData,t, h, dz_req):
    from scipy.interpolate import RegularGridInterpolator

    z,zmax,zmin = profileData.z, profileData.zmax , profileData.zmin
    time_given = timeData.time_given

    Z, T = np.meshgrid(z, t, indexing='ij')
    interp = RegularGridInterpolator((z, t), h, bounds_error=False, fill_value=None)
    z_req = np.arange ( zmin,zmax,dz_req)
    t_req = time_given
    Z_req, T_req = np.meshgrid(z_req, t_req, indexing='ij')
    # RegularGridInterpolator expects points as (N, 2)
    pts = np.column_stack([Z_req.ravel(), T_req.ravel()])
    h_req = interp(pts).reshape(Z_req.shape)  # same shape as X_req/Y_req
    depth_req = zmax - z_req
    
    return h_req,z_req, depth_req 



def prcoessed_results(profileData,timeData,SoilData, t, h, dz_req):
    h_req,z_req, depth_req  = spatial_interpolation(profileData,timeData,t, h, dz_req)
    
    theta_req = np.empty(h_req.shape)
    Se_req = np.empty(h_req.shape)
    vgData_req = assign_vg(depth_req, SoilData,profileData.zmax)    

    nodes_1m_depth = depth_req<=1
    nodes_10cm_depth = depth_req<=0.1
    water_storage_100_cm_req = np.empty(len(timeData.time_given))
    water_storage_10_cm_req = np.empty(len(timeData.time_given))
    for i in range(len(timeData.time_given)):
        theta_req[:,i], _,  _, _ = vgModel( h_req[:,i] , vgData_req )
        Se_req[:,i]  = ( theta_req[:,i] - vgData_req['thetar'] ) / (vgData_req['thetas'] -vgData_req['thetar'] )
        
        water_storage_10_cm_req[i] = np.trapezoid(theta_req[nodes_10cm_depth,i], x=z_req[nodes_10cm_depth])*1000
        water_storage_100_cm_req[i] = np.trapezoid(theta_req[nodes_1m_depth,i], x=z_req[nodes_1m_depth])*1000  
    
    return theta_req,Se_req, h_req, z_req, depth_req , water_storage_10_cm_req, water_storage_100_cm_req




































# def ponding_flux(h,Ksat, thetar, thetas , alpha, N, n_eta,dz_all,z):
#     # estimation of flux at the top when ponding condition has been reached, i.e., Se >= 0.999 
#     Se_top = 0.999
    
#     # code to estimate the pressure head the point 
#     znodes = len(dz_all)
    
#     i = znodes-1
#     thetas_current = thetas[i]
#     thetar_current = thetar[i]
#     alpha_current = alpha[i]
#     N_current = N[i]
#     Ksat_current = Ksat[i]
#     n_eta_current = n_eta[i]

  
#     theta_top = Se_top*(thetas_current - thetar_current) + thetar_current
#     m = 1-1/N_current
    
#     K_top = Ksat_current*(Se_top**n_eta_current)*( ( 1 - (1 - Se_top**(1/m))**m  )**2)

#     f = lambda h: Van_Genuchten_moisure(h, thetar_current, thetas_current, alpha_current, N_current) - theta_top

#     # # Initial guess (h must be negative in unsaturated soil)
#     h_top  = fsolve(f, x0=-1)[0]
#     # K_top = Van_Genuchten_K_alt(h_top,Ksat, alpha, N, n_eta)
  
#     h2  = h[i]
#     K2 = Van_Genuchten_K( h2 , Ksat_current, thetar_current, thetas_current , alpha_current, N_current, n_eta_current)
    
#     d2z = dz_all[i]
  
#     Kszminus = ( K2 + K_top )  / 2 
#     # Kszminus = ( K2 )
#     # Kszminus = ( K_top )


#     q = -Kszminus*( ( h2- h_top)/(d2z/2) -1  )
        
    
#     return q

































