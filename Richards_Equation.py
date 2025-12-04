
"""
@author: Saurabh

Code based on Dogan and Mortz 2002a,b 
# Dogan, A., & Motz, L. H. (2005). Saturated-unsaturated 3D groundwater model. I: Development. 
# Dogan, A., & Motz, L. H. (2005). Saturated-unsaturated 3D groundwater model. II: Verification and application
# We are using SI units but except of seconds is replaced by day
h is in m 
Ksat = m/day


"""
# input modules for the code 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass, field
import time
from vg_models import vgModel
from re_model_function_files import Richards_Solver,soil_grid,assign_vg, prcoessed_results
t0= time.perf_counter()
# importing meterological data from .xlsx files dataframa contains, crop  transpiration, crop evaporation, crop ET, rain in mm/day 
MetData = pd.read_excel('data_johnstown.xlsx',sheet_name='met_data')  
# soil grid data and code for soil profile generation
@dataclass
class ProfileGridSpec:
    zmin: float
    zmax: float
    dz: float                 # requested min spacing
    z: np.ndarray = field(init=False)
    dz_all: float = field(init=False)
    depth: int = field(init=False)
    
    def __post_init__(self):
        self.z, self.dz_all, self.depth = soil_grid(self.zmax, self.dz)
        
profileData = ProfileGridSpec( zmin=0, zmax=1.8, dz=0.006 )

# initial condition 
h_i_obs_hpa = np.array([60,45,1,13,49,70 , 70])
h_i_obs_m  = - 1.0197*h_i_obs_hpa/100
depth_obs = [0.15,0.3,0.45,0.60,0.90,1.20, 3.0]
h_ini = pd.DataFrame({
    'depth': depth_obs,
    'h': h_i_obs_m,
    'h_hpa': h_i_obs_hpa
})


# SHP data
SoilData = pd.read_excel('data_johnstown.xlsx',sheet_name='depthAvgVGRosetta')
vgData = assign_vg(profileData.depth, SoilData,profileData.zmax)    


# Root water uptake parameters 
@dataclass

class RWUSpec:
    psi_a: float        # critical pressure heads associated with anaerobiosis,
    psi_d: float        #critical pressure heads associated with soilwater-limited evapotranspiration
    psi_w: float      # critical pressure head associated with plant wilting
    Lr: float      # max root length (uniform grid)

RWUData = RWUSpec(
    psi_a=-0.25,  # critical pressure heads associated with anaerobiosis,
    psi_d=-3,   # critical pressure heads associated with soilwater-limited evapotranspiration
    psi_w=-10,  # # critical pressure head associated with plant wilting
    Lr=0.5  ) # m # depth of root zone

# run time details and interval  

@dataclass
class TimeSpec:
    tmin: float # start time 
    tmax: float # finsh time length 
    dt_min: float # min time step  
    dt_max: float # max time step 
    dt : float  #in day 
    plus_factor: float # factor with which time step dt will  increase 
    dec_factor: float  # decrease factor for time step dt
    time_given: float # daily time 
    
timeData = TimeSpec(
    tmin = 1,
    tmax = len(MetData),
    dt_min= 0.0000000000000000000000000000000001,
    dt_max=1,
    dt = 1 ,  #in day 
    plus_factor=0.1,
    dec_factor=0.3,
    time_given = np.arange(1,len(MetData)+1)
)
# running the code 
theta, K, h, Se,Ea,RWU,t, Ta, Error,q_ponding_flux,q_rain =  Richards_Solver(profileData, timeData,vgData,MetData,RWUData,h_ini )
h_hPa = h*100/1.0197  # obtaining values of water tenion values in hPa

t1 = time.perf_counter() - t0
print("Time elapsed: ", t1) # CPU seconds elapsed 

dz_req = 0.005
theta_req,Se_req, h_req, z_req, depth_req , water_storage_10_cm_req, water_storage_100_cm_req = prcoessed_results(profileData,timeData,SoilData, t, h, dz_req)

start_date = "1998-01-01"
end_date = "2001-01-01"  # Adjust if you get data beyond 1995

time_idx = (MetData["date"] >= start_date) & (MetData["date"] <= end_date)
time_idx = time_idx.to_numpy()

theta_1998_2000 = theta_req[ :,time_idx]

Se_1998_2000 =  Se_req[ :,time_idx]

water_storage_10_cm_req_1998_2000 = water_storage_10_cm_req[time_idx]
water_storage_100_cm_req_1998_2000 =water_storage_100_cm_req[time_idx]

MetData['water_storage_10cm'] = water_storage_10_cm_req
MetData['water_storage_100cm'] = water_storage_100_cm_req

# values of depths at knows comparison will be done 
depth_vals_plot = np.array([ 0.10, 0.15,0.45,1.20])
node_indices = [np.argmin(np.abs(depth_req - d)) for d in depth_vals_plot]
node_indices = np.array(node_indices)

MetData['h_10cm'] = h_req [node_indices[0],:] 
MetData['h_15cm'] = h_req [node_indices[1],:] 
MetData['h_45cm'] = h_req[node_indices[2],:] 
MetData['h_120cm'] = h_req[node_indices[3],:] 

MetData['theta_10cm'] = theta_req [node_indices[0],:] 
MetData['theta_15cm'] = theta_req [node_indices[1],:] 
MetData['theta_45cm'] = theta_req[node_indices[2],:] 
MetData['theta_120cm'] = theta_req[node_indices[3],:] 

MetData['Se_10cm'] = Se_req [node_indices[0],:] 
MetData['Se_15cm'] = Se_req [node_indices[1],:] 
MetData['Se_45cm'] = Se_req[node_indices[2],:] 
MetData['Se_120cm'] = Se_req[node_indices[3],:] 

   
df_filtered = MetData[(MetData["date"] >= start_date) & (MetData["date"] <= end_date)]

obs_data_15_cm = pd.read_excel('data_johnstown.xlsx',sheet_name='obs_data_15_cm')
obs_data_45_cm = pd.read_excel('data_johnstown.xlsx',sheet_name='obs_data_45_cm')
obs_data_120_cm = pd.read_excel('data_johnstown.xlsx',sheet_name='obs_data_120_cm')



fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

# --- 15 cm ---
axes[0].plot(df_filtered["date"], -(df_filtered['h_15cm'])*100/1.0197, label='pred')
axes[0].plot(obs_data_15_cm["Date"], obs_data_15_cm['Tension_value_hPa'], '.', label='obs')
axes[0].set_ylabel(r"Tension value (hPa)")
axes[0].grid()
axes[0].set_ylim([-250,1000])
axes[0].legend()
axes[0].set_title('a) Soil water tension at 15 cm depth')

# --- 45 cm ---
axes[1].plot(df_filtered["date"], -(df_filtered['h_45cm'])*100/1.0197 , label='pred')
axes[1].plot(obs_data_45_cm["Date"], obs_data_45_cm['Tension_value_hPa'], '.', label='obs')
axes[1].set_ylabel(r" Tension value (hPa)")
axes[1].grid()
# axes[1].legend()
axes[1].set_ylim([-250,1000])

axes[1].set_title('b) Soil water tension at 45 cm depth')

# --- 120 cm ---
axes[2].plot(df_filtered["date"], -(df_filtered['h_120cm'])*100/1.0197 , label='pred')
axes[2].plot(obs_data_120_cm["Date"], obs_data_120_cm['Tension_value_hPa'], '.', label='obs')
axes[2].set_ylabel(r"Tension value (hPa)")
axes[2].grid()
# axes[2].legend()
axes[2].set_title('c) Soil water tension at 120 cm depth')
axes[2].set_ylim([-250,1000])

# --- Common labels ---
axes[2].set_xlabel("Date")
# fig.suptitle("Variation of Soil Pressure Head with Time", fontsize=14)
fig.tight_layout()

plt.savefig('comparison_water_tension_Johncastle_depthavgRosetta_fixedHead.svg', dpi = 300)
plt.savefig('comparison_water_tension_Johncastle_depthavgRosetta_fixedHead.png', dpi = 300)


plt.show()

fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

# --- 15 cm ---
axes[0].plot(df_filtered["date"], abs(df_filtered['theta_15cm']), label='pred')
axes[0].plot(obs_data_15_cm["Date"], obs_data_15_cm['theta'], '.', label='obs')
axes[0].set_ylabel(r"$\theta $")
axes[0].grid()
axes[0].set_ylim([0.1,0.5])
axes[0].legend()
axes[0].set_title('a) Soil moisture at 15 cm depth')

# --- 45 cm ---
axes[1].plot(df_filtered["date"], abs(df_filtered['theta_45cm']), label='depth = 45 cm')
axes[1].plot(obs_data_45_cm["Date"], obs_data_45_cm['theta'], '.', label='obs')
axes[1].set_ylabel(r"$\theta $")
axes[1].grid()
# axes[1].legend()
axes[1].set_ylim([0.1,0.5])

axes[1].set_title('b) Soil moisture at 45 cm depth')

# --- 120 cm ---
axes[2].plot(df_filtered["date"], abs(df_filtered['theta_120cm']), label='depth = 120 cm')
axes[2].plot(obs_data_120_cm["Date"], obs_data_120_cm['theta'], '.', label='obs')
axes[2].set_ylabel(r"$\theta $")
axes[2].grid()
# axes[2].legend()
axes[2].set_title('c) Soil moisture at 120 cm depth')
axes[2].set_ylim([0.1,0.5])

# --- Common labels ---
axes[2].set_xlabel("Date")
# fig.suptitle("Variation of Soil Pressure Head with Time", fontsize=14)
fig.tight_layout()

plt.savefig('comparison_soil_moisture_Johncastle_depthavgRosetta_fixedHead.svg', dpi = 300)
plt.savefig('comparison_soil_moisture_Johncastle_depthavgRosetta_fixedHead.png', dpi = 300)

plt.show()







# plt.figure(figsize=(8, 4) )
# plt.plot(t, -Ea, label= 'Ea')
# plt.plot(t, Ta, label= 'Ta')
# plt.plot(timeData.time_given, PET*1000, label= 'pet')

# plt.ylabel(r"$E_a $ and $T_a$")
# plt.grid()
# # plt.ylim([0.25, 0.40])
# plt.legend()
# plt.title('Variation of soil moisture with time')
# plt.show()


# plt.figure(figsize=(10, 4) )
# plt.plot(t, Se[znodes-1,:], label='depth = 0.003 m')
# plt.grid()
# plt.ylabel(r"$S_e$")
# plt.xlabel(r"$Days$")
# plt.title('Variation of Se at surface for entire simulation runtime 1995 -2001')
# plt.show()


plt.figure(figsize=(10, 4) )
plt.plot(df_filtered["date"], abs(df_filtered['Se_15cm']), label='depth = 15 cm')
plt.plot(df_filtered["date"], abs(df_filtered['Se_45cm']), label='depth = 45 cm')
plt.plot(df_filtered["date"], abs(df_filtered['Se_120cm']), label='depth = 120 cm')
plt.ylabel(r"$\theta $")
plt.grid()
plt.ylim([0, 1])
plt.legend()
plt.title('Variation of degree of saturation with time')

# --- Common x-axis formatting ---
plt.xticks(pd.date_range(start=start_date, end=end_date, freq="3MS"), rotation=45)
plt.tight_layout()
# plt.savefig("soil_tensiometer_head.png",dpi = 300)
# plt.savefig("soil_tensiometer_head.svg",dpi = 300)
plt.show()


# # # ------------------------ create a animation of theta and saturation 

# ------------------ DEMO WITH SYNTHETIC DATA ------------------
# Create a synthetic theta(z, t) to showcase the animation:
# A moistening front moving downward with gentle sinusoidal fluctuation over time.
nz, nt = theta_1998_2000.shape

t_1999_2000 = np.arange(1,nt+1,1)
# z_req = np.arange ( zmin,zmax,0.05)
# depth_req = zmax - z_req
# z = np.linspace(zmin, zmax, nz)           # e.g., meters, 0 at surface -> 1 m depth
# filename="theta_15m_fixed_head.gif" 
filename="theta_1m_free_drainage.gif" 

# out_path = animate_profile(theta_1999_2000, depth_req, t_1999_2000, filename, interval_ms=3000, dpi=200)
# out_path
# filename="Se_15m_fixed_head.gif" 
filename="Se_1m_free_drainage.gif" 

# out_path = animate_profile_Se(Se_1999_2000, depth_req, t_1999_2000, filename, interval_ms=3000, dpi=200)



# # mat_data = {'Error':Error,
# #             'theta':theta,
# #             'h':h,
# #             't':t,
# #             'theta_new':theta_new,
# #             'MetData':MetData ,
# #             'SoilData':SoilData,
# #             'profileData': profileData,
# #             }

# # scipy.io.savemat('results_hohncastle.mat', mat_data)











    
 