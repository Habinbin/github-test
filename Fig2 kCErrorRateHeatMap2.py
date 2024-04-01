import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import warnings
import proplot as pplt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from scipy.interpolate import interp2d
warnings.filterwarnings("ignore")

def cm2in(cm):
    return cm/2.54

def arr2df(arr):
    return pd.DataFrame(arr)

# font
subplot_no = list(map(chr,range(97,123)))

# font
fs = 10
FS_label = fs+1
Axis_FS =fs-1
LEG_FS = fs-2
Annotation_FS = fs-0.5

# unit change 
day_to_hour = 24
hour_to_min = 60
hour_to_sec = 3600
min_to_sec = 60  
cm2m = 1/100
MJ2J = 10**6


# read the csv file
Z = np.array(pd.read_csv("kCError 5 cm 50x50(log).csv")) # read the csv file
Z = np.array(pd.read_csv("kCError 5 cm.csv"))
Z1 = pd.read_csv("kCError 5 cm 50x50(log).csv") # read the csv file
Z2 = Z1.iloc[:10,:10]     # k:0.01 , C:10^4
Z3 = Z1.iloc[10:20,10:20] # k:0.1  , C:10^5
Z4 = Z1.iloc[20:30,20:30] # k:1    , C:10^6
# # Interpolated Z
# row_ip = [10,100,1000,10000]
# col_ip = [10,100,1000]


# # Volumetric Heat capcity [J/m^3K] (x-coordinate)
# C_arr = np.concatenate((np.linspace(0.01,0.1,9,endpoint=False)*MJ2J 
# ,np.linspace(0.01,0.1,9,endpoint=False)*10*MJ2J 
# ,np.linspace(0.01,0.1,10)*100*MJ2J 
# )) 
# NC = len(C_arr) 
# C_arr[9]

# # System thermal conductivity [W/mK] (y-coordinate)
# k_arr = np.concatenate((np.linspace(0.01,0.1,9,endpoint=False) 
# ,np.linspace(0.01,0.1,9,endpoint=False)*10
# ,np.linspace(0.01,0.1,9,endpoint=False)*100
# ,np.linspace(0.01,0.1,9,endpoint=False)*1000
# ,np.linspace(0.01,0.1,10)*10000))
# Nk = len(k_arr) 

# num_rows = 4
# num_cols = 3
# array_list = [[None] * num_cols for _ in range(num_rows)]
# array_list

# for ridx in row_ip:
#     for cidx in col_ip:
#         i = row_ip.index(ridx)
#         j = col_ip.index(cidx) 
#         Zij = Z[10*i:10*(i+1),10*j:10*(j+1)] 

#         x = np.linspace(10**(j+4),10**(j+5),Zij.shape[1])
#         y = np.linspace(0.01*10**(i),0.01*10**(i+1),Zij.shape[0])

#         X,Y = np.meshgrid(x, y)
#         f = interp2d(x, y, Zij, kind='linear')

#         x_dens = np.linspace(x.min(), x.max(), cidx)
#         y_dens = np.linspace(y.min(), y.max(), ridx)

#         X_dens,Y_dens = np.meshgrid(x_dens, y_dens)

#         Zij_dens = f(x_dens,y_dens)
#         array_list[i][j]=Zij_dens

# result_array = np.concatenate([np.concatenate(sublist, axis=1) for sublist in array_list], axis=0)
# arr2df(result_array)


# figure settings
min_k = 0.01
max_k = 1000

min_C = 10**4
max_C = 10**7

norm1 = mcolors.Normalize(vmin = 0, vmax = 90)

# figure settings
nrow = 1
ncol = 1
nfigs = nrow*ncol


fig, ax = plt.subplots(nrow, ncol, sharex=False, sharey=False, figsize=(cm2in(19),cm2in(15)), 
                        facecolor='w', edgecolor='k', squeeze=False, dpi = 300)


xmin = [min_C]*nfigs
xmax = [max_C]*nfigs
xmar = [0]*nfigs

ymin = [min_k]*nfigs
ymax = [max_k]*nfigs 
ymar = [0]*nfigs # yint[i]/6

# figure settings 
for ridx in range(nrow):
    for cidx in range(ncol):
        idx = ridx*ncol + cidx

        Z = Z #Z_lst[idx]

        x = np.linspace(min_C,max_C,Z.shape[1])
        y = np.linspace(min_k,max_k,Z.shape[0])

        X,Y = np.meshgrid(x, y)

        # 2D interpolation
        f = interp2d(x, y, Z, kind='linear')

        # x, y grid 
        x_dens = np.linspace(x.min(), x.max(), 1000)
        y_dens = np.linspace(y.min(), y.max(), 10000)

        X_dens,Y_dens = np.meshgrid(x_dens, y_dens)

        Z_dens =f(x_dens,y_dens)

        ef1 = ax[0,0].imshow(Z, aspect='auto', extent=[min_C,max_C,max_k,min_k], 
                            cmap='Blues9', norm=norm1,) # Error field  interpolation = "spline16" 

        ax[ridx,cidx].set_xlabel('Volumetric heat capacity [J/m3K]', fontsize=FS_label,labelpad=10)
        ax[0,0].set_ylabel('Thermal conductivity [10$^{-6}$ m$^2$/s]', fontsize=FS_label,labelpad=10)

        ax[ridx,cidx].tick_params(direction='in', labelsize=Axis_FS, which='major', length=2.5, width=0.5 ,  right=True,top=True ,pad=6)
        ax[ridx,cidx].tick_params(direction='in', labelsize=Axis_FS, which='minor', length=1.25, width=0.25, right=True,top=True ,pad=6)

        # xlim ylim 
        ax[ridx,cidx].set_xlim(xmin[idx]-xmar[idx], xmax[idx]+xmar[idx])
        ax[ridx,cidx].set_ylim(ymin[idx]-ymar[idx], ymax[idx]+ymar[idx])     
        
        # ax[ridx,cidx].set_xticks(np.arange(xmin[idx], xmax[idx]+xint[idx], xint[idx]))
        # xtick = np.arange(min_k, max_k+k_int, k_int*4).tolist()
        # ytick = np.arange(min_C, max_C+C_int, C_int*4).tolist()

        # xticks, yticks
        # ax[ridx,cidx].set_xticks([round(xtick[i],2) for i in range(len(xtick))]) 
        # ax[ridx,cidx].set_yticks([round(ytick[i],2) for i in range(len(ytick))])
        
        # replace xtick string 
        # xtick_list = ax[ridx,cidx].get_xticks().tolist()
        # ax[ridx,cidx].set_xticklabels(xtick_list)
        # ytick_list = ax[ridx,cidx].get_yticks().tolist()
        # ax[ridx,cidx].set_yticklabels(ytick_list)
        
        # number of minor ticks
        ax[ridx,cidx].xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax[ridx,cidx].yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

        # grid
        ax[ridx,cidx].grid(True, axis='both', linestyle=':', linewidth=0.5, color='k'); 
        # subplot
        # subplot_idx = '('+subplot_no[idx]+') ' + case_name[idx] #(a) + Internal
        #ax[ridx,cidx].annotate(subplot_idx, xy=(.01, 1.03), xycoords='axes fraction',
            #horizontalalignment='left', verticalalignment='top', fontsize=fs) 
        
        # scale
        ax[ridx,cidx].set_xscale('log')
        ax[ridx,cidx].set_yscale('log')

        # 등고선 그리기
        contour_levels = np.array([int(5)]+ [int(10*(i+1)) for i in range(8)]) # couter1
        # contour = ax[ridx,cidx].contour(X_dens, Y_dens, Z_dens, levels=contour_levels, colors='0.1', linewidths=0.75, linestyles='-') # denser version
        contour = ax[ridx,cidx].contour(X, Y, Z, levels=contour_levels, colors='0.1', linewidths=0.75, linestyles='-') # normal version
        ax[ridx,cidx].clabel(contour, fontsize=fs-2, fmt = '%1.0f')
                 
        # spine line width  
        for k in ['top','bottom','left','right']:
                ax[ridx,cidx].spines[k].set_visible(True)
                ax[ridx,cidx].spines[k].set_linewidth(0.5)
                ax[ridx,cidx].spines[k].set_color('k')         
plt.tight_layout(pad = 1.7)
fig.subplots_adjust(right=0.85, bottom=0.12) # make space for additional colorbar

# colorbar
cbar_width =  0.015 #vertical  
cbar_height = 0.015 #horizontal  
cbar_dist_v = 0.085; # vertical colorbar distance from bbox edge
cbar_dist_h = 0.028; # horizontal colorbar distance from bbox edge

# horizontal colorbars 
# colorbar1 temperature 
bbox = ax[0,0].get_position() # get the normalized axis original position 
cb_ax = fig.add_axes([bbox.x1+cbar_dist_h, bbox.y0, cbar_width, bbox.y1-bbox.y0]) #[x_origin, y_origin, width, height]
cbar1  = fig.colorbar(ef1, cax=cb_ax, ax=ax[0,0], orientation='vertical') 
cbar1.ax.tick_params(direction='in',labelsize=Axis_FS, length=2, width=0.5, pad=3)
cbar1.ax.minorticks_off()
cbar1.locator = ticker.MultipleLocator(10)
cbar1.ax.set_ylabel('Percentage error [%]', rotation=90, fontsize=FS_label, labelpad=10)
cbar1.outline.set_linewidth(0.5)

figname = 'Fig2 Percentage error Comparison'
# plt.savefig(figname+'.svg', dpi=600)
# plt.savefig(figname+'.png', dpi=600)
# plt.savefig(figname+'.pdf', dpi=600)



