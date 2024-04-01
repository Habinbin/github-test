import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

# Layer making class
class SetLayer:
    def __init__(self, L, dx, k, C):
        self.L = L # Length [m]
        self.dx = dx # dx [m]
        self.k = k # Thermal conductivity [W/mK]
        # self.rho = rho # Density [kg/m^3]
        # self.c = c # Specific heat capacity [J/kgK]
        self.C = C # Volumetric heat capacity [J/m^3K]
        self.div = int(L/dx) # Number of division [-] 
        self.R = dx/k # Thermal resistance [m^2K/W]
        self.K = k/dx # Thermal conductance # [W/m^2K]
        self.alpha = k/C # Thermal diffusivity [m^2/s]

# Construction making class
class SetConstruction:
    def __init__(self, *layers):
        self.layers = layers
        self.N = sum([Lidx.div for Lidx in layers])
        self.construction = list(layers)

    def dx(self):
        N = self.N
        Construction = self.construction
        dx =   np.array([Lidx.dx for Lidx in Construction for _ in range(N)])  #C length [m]
        dx_L = np.array([dx[0]/2]+[(dx[i-1] + dx[i])/2 for i in range(1,N)])   #Left interface C length
        dx_R = np.array([(dx[i] + dx[i+1])/2 for i in range(N-1)]+[dx[N-1]/2]) #Right interface C length
        return dx, dx_L, dx_R

    def K(self):
        N = self.N
        Construction = self.construction
        R = np.array([Lidx.R for Lidx in Construction for _ in range(N)]) #Thermal resistance [m^2K/W]
        K = 1/R #Thermal conductance [W/m^2K]  
        K_L = 1/np.array([R[0]/2]+[(R[i-1] + R[i])/2 for i in range(1,N)]) #Left interface thermal conductance [W/m^2K]
        K_R = 1/np.array([(R[i] + R[i+1])/2 for i in range(N-1)]+[R[N-1]/2]) #Right interface thermal conductance [W/m^2K]
        return K, K_L, K_R
    
    def R(self):
        N = self.N
        Construction = self.construction
        R = np.array([Lidx.R for Lidx in Construction for _ in range(N)]) #Thermal resistance [m^2K/W]
        return R
    
    def K_tot(self):
        N = self.N
        Construction = self.construction
        R_tot = sum([Lidx.R for Lidx in Construction for _ in range(N)]) #Total thermal resistance [m^2K/W]
        K_tot = 1/R_tot
        return K_tot
    
    def R_tot(self):
        N = self.N
        Construction = self.construction
        R_tot = sum([Lidx.R for Lidx in Construction for _ in range(N)])
        return R_tot
    
    def C(self):
        N = self.N
        Construction = self.construction
        C = np.array([Lidx.C for Lidx in Construction for _ in range(N)]) #Volumetric heat capacity [J/m^3K]
        return C

# Function
def D2K(Degree): #Degree to Kelvin
    Kelvin = Degree + 273.15
    return Kelvin

def K2D(Kelvin): #Kelvin to Degree
    Degree = Kelvin - 273.15
    return Degree

def arr2df(arr): # Array to DataFrame
    df = pd.DataFrame(arr)
    return df

def cm2in(cm): # cm to inch
    inch = cm/2.54
    return inch

def hftlist(lst): # Half time step list
    hflst = [(lst[i]+lst[i+1])/2 for i in range(len(lst)-1)]
    return hflst

def Aver2col(arr): # Average to column side
    col = arr.shape[1] # Number of row(time step)
    colAverArr = np.array([(arr[:,i]+arr[:,i+1])/2 for i in range(col-1)]).T
    return colAverArr # column averaged array

def Aver2row(arr): # Average to row side
    row = arr.shape[0] # Number of rows(time step)
    arrInv = arr.T
    rowAverArr = np.array([(arrInv[:,i]+arrInv[:,i+1])/2 for i in range(row-1)])
    return rowAverArr # row averaged array


# Unit change 
d2h = 24
h2m = 60
h2s = 3600
s2h = 1/3600
m2s = 60  
s2m = 1/60
m2cm = 100
MJ2J = 10**6
J2MJ = 1/10**6
J2kJ = 1/10**3
kJ2J = 10**3


# Time variable
dt = 10 #[s] 
PST = 120 #PS:Pre Simulation[h]
EST = 144 #TS:Total Simulation[h]

Nt = int(EST*h2s/dt) # Number of time steps 
dt_lst_sec = np.array([dt*i for i in range(Nt+1)]) #time step list [t1,t2,..., tN]
dt_lst_min = dt_lst_sec*s2m; #minute
dt_lst_hour = dt_lst_sec*s2h; #hour

N_PST = int(PST*h2s/dt) # Number of pre-simulation time steps
N_EST = int(EST*h2s/dt) # Number of total simulation time steps


# Calculation condition
T_init = D2K(20) 
LBC, RBC = np.zeros((Nt+1,1)), np.zeros((Nt+1,1)) #Left boundary condition, Right boundary condition
LBC[:,0] = np.array([(D2K(20 + 10*(math.sin(2*math.pi*n/d2h)))) for n in dt_lst_hour]) #Left boundary condition
RBC[:,0] = np.array([(D2K(20)) for _ in dt_lst_hour]) # Right boundary condition
T0 = LBC.copy() #Environment temperature


# System length [m]
L_arr = np.array([0.1])
NL = len(L_arr) # Number of length

# Volumetric Heat capcity [J/m^3K] (x-coordinate)
C_arr = np.linspace(2*10**6,3*10**6,10,endpoint=False)
C_arr
NC = len(C_arr) 
# System thermal conductivity [W/mK] (y-coordinate)
k_arr = np.linspace(0.01,100,10000)
Nk = len(k_arr) 

# DataFrame to save
x = C_arr
y = k_arr
x_pos, y_pos = np.meshgrid(x, y)

# Run
for Lidx in range(NL): #System length(x axis)
    PEMatrix = np.zeros((Nk,NC))
    FileName = f"kCError2 {int(L_arr[Lidx]*m2cm)} cm.csv"
    for Cidx in tqdm(range(NC)): #Volumetric heat capacity (x axis)
        for kidx in range(Nk): #Thermal conductivity (y axis)
            # Define the thermal network
            Layer = SetLayer(L = L_arr[Lidx], 
                            dx = 0.01,
                            k=k_arr[kidx],
                            C=C_arr[Cidx],
            ) 
            Construction = SetConstruction(Layer)

            N = Construction.N #Number of nodes

            # Define the thermal network
            dx, dx_L, dx_R = Construction.dx()
            K, K_L, K_R = Construction.K()
            C = Construction.C()
            rows, cols = Nt+1, N # Number of rows(time step), Number of columns(node)

            T_origin = np.full((rows,cols),T_init) #Initial temperature
            q_origin = np.zeros((rows,cols)) #Initial heatflux
            Carneff_origin = np.zeros((rows,cols)) #Initial Carnot coefficient

            T_L, T, T_R = T_origin.copy(), T_origin.copy(), T_origin.copy() #Temperature
            q_in, q, q_out = q_origin.copy(), q_origin.copy(), q_origin.copy() #Heatflux
            T_L_hf, T_hf, T_R_hf = T_origin.copy(), T_origin.copy(), T_origin.copy() #Half time step temperature
            Carneff = Carneff_origin.copy() #Carnot coefficient

            # Initial Temperature, boundary condition
            T_L[:,0], T_R[:,N-1] = LBC[:,0], RBC[:,0] # Left and right boundary condition

            # TDMA(Triangle Diagonal mat Algorithm)
            aList = -dt*K_L
            bList = 2*dx*C+dt*K_L+dt*K_R
            cList = -dt*K_R

            # Amat*T = Bmat
            Amat = np.zeros((N,N))
            Bmat = np.zeros((N,1))
            for i in range(N-1):
                Amat[i+1,i]   = aList[i+1]
                Amat[i,i]     = bList[i]
                Amat[N-1,N-1] = bList[N-1]
                Amat[i,i+1]   = cList[i]
                
            AmatInv = np.linalg.inv(Amat)

            for n in range(Nt):
                Bmat[0,0] = 2*dt*K_L[0]*T_L[n,0]+(2*dx[0]*C[0]-dt*K_L[0]-dt*K_R[0])*T[n,0]+dt*K_R[0]*T[n,1] # g1
                Bmat[1:N-1,0] = dt*K_L[1:N-1]*T[n,0:N-2]+(2*dx[1:N-1]*C[1:N-1]-dt*K_L[1:N-1]-dt*K_R[1:N-1])*T[n,1:N-1]+dt*K_R[1:N-1]*T[n,2:N]
                Bmat[N-1,0] = dt*K_L[N-1]*T[n,N-2]+(2*dx[N-1]*C[N-1]-dt*K_L[N-1]-dt*K_R[N-1])*T[n,N-1]+2*dt*K_R[N-1]*T_R[n,N-1] # gN
                
                T[n+1,:] = np.dot(AmatInv, Bmat)[:,0]
                T_L[n+1,1:N] = np.array([(T[n+1,i-1]+T[n+1,i])/2 for i in range(1,N)])
                T_R[n+1,:N-1] = np.array([(T[n+1,i]+T[n+1,i+1])/2 for i in range(N-1)])

            # Heatflux
            q_in[:,1:N] = K_L[1:N]*(T[:,0:N-1] - T[:,1:N])
            q_in[:,0] = K_L[0]*(T_L[:,0]-T[:,0]) #Heat influx 1st half node
            q_out[:,0:N-1] = K_R[0:N-1]*(T[:,0:N-1] - T[:,1:N])
            q_out[:,N-1] = K_R[N-1]*(T[:,N-1]-T_R[:,N-1]) #Heat outflux last half node
            q[:,:] = (q_in[:,:] + q_out[:,:])/2

            # Define half time step values
            T_L_hf = Aver2row(T_L)
            T_hf = Aver2row(T)
            T_R_hf = Aver2row(T_R)

            T0_hf = Aver2row(T0)

            q_in_hf = Aver2row(q_in)
            q_hf = Aver2row(q)
            q_out_hf = Aver2row(q_out)

            # Carnot coefficient 
            Carneff_L = 1-(T0/T_L)
            Carneff_R = 1-(T0/T_R)

            # Define unsteady-state exergy values (Unsteady-state)
            U_CXifR = q_in_hf*(1-(T0_hf/T_L_hf)) #Exergy inflow R
            U_CXcR= (1/K)*(T0_hf*(q_hf/T_hf)**2) #Exergy consumption R
            U_CXstR = dx*C*(1-T0_hf/T_hf)*((T[1:Nt+1,:]-T[0:Nt,:])/dt) #Exergy stored R
            U_CXofR = q_out_hf*(1-T0_hf/T_R_hf) #Exergy outflow Rate
            U_CXst = C*dx*((T-T0)-T0*np.log(T/T0)) #Stored exergy in Cell

            # Define steady-state exergy values (Dynamic)
            q_tot = Construction.K_tot()*(LBC - RBC) #Total heat flux 
            D_XifR  = (1-(T0/LBC))*q_tot #Exergy inflow R
            D_XcR  = Construction.R_tot()*(q_tot**2/(LBC*RBC))*T0 #Exergy consumption R
            D_XofR = (1-(T0/RBC))*q_tot #Exergy outflow R
            
            # Exergy
            U_CXifR = U_CXifR[N_PST:,:] 
            U_CXcR = U_CXcR[N_PST:,:]
            U_XcR = np.sum(U_CXcR,axis=1) #Spacially integrated exergy consumption
            U_Xc = np.sum(U_XcR)*dt #Time integrated exergy consumption
            U_CXstR = U_CXstR[N_PST:,:]
            U_CXofR = U_CXofR[N_PST:,:]
            U_CXst = U_CXst[N_PST:,:]

            D_XifR = D_XifR[N_PST:N_EST-1] 
            D_XcR = D_XcR[N_PST:N_EST-1] 
            D_Xc = np.sum(D_XcR,axis=0)*dt 
            D_XofR = D_XofR[N_PST:N_EST-1]

            # Percentage error
            Error = abs((U_Xc-D_Xc)/U_Xc)*100 #Exergy consumption percentage error
            PEMatrix[kidx,Cidx] = Error #Save in dataframe
    arr2df(PEMatrix).to_csv(f"{FileName}", index=False)
