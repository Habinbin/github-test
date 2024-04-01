import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

def cm2in(cm):
    return cm/2.54

fs = 10

MJ2J = 1e6
J2MJ = 1e-6

xmax = 5*MJ2J # Volumetric heat capacity [MJ/m3K]
xmin = 0.1*MJ2J

ymax = 100 # Thermal conductivity [W/mK]
ymin = 0.01

x = np.linspace(xmin, xmax, 10)
y = np.linspace(ymin, ymax, 10)
alp_arr = np.array([8e-8, 2e-7,2e-6, 5e-6, 2e-5, 5e-5])
Nalp = alp_arr.shape[0]
y_arr = [alp_arr[i]*x for i in range(Nalp)]

nrow = 1
ncol = 1
nfigs = nrow*ncol 

xmin = [xmin]*nfigs
xmax = [xmax]*nfigs

ymin = [ymin]*nfigs
ymax = [ymax]*nfigs 

# figure settings
fig, ax = plt.subplots(nrow, ncol, sharex=False, sharey=False, figsize=(cm2in(19),cm2in(15)), 
                        facecolor='w', edgecolor='k', squeeze=False, dpi = 300)

# figure settings 
for ridx in range(nrow):
    for cidx in range(ncol):
        for nidx in range(Nalp):

            idx = ridx*ncol + cidx

            ax[ridx,cidx].plot(x, y_arr[nidx], label = alp_arr.astype(str)[nidx])
            
            ax[ridx,cidx].set_xlabel('Volumetric heat capacity [MJ/m3K]', fontsize=fs,labelpad=10)
            ax[ridx,cidx].set_ylabel('Thermal conductivity [W/mK]', fontsize=fs,labelpad=10)

            ax[ridx,cidx].tick_params(direction='in', labelsize=fs, which='major', length=2.5, width=0.5 ,  right=True,top=True ,pad=6)
            ax[ridx,cidx].tick_params(direction='in', labelsize=fs, which='minor', length=1.25, width=0.25, right=True,top=True ,pad=6)

            # xlim ylim 
            ax[ridx,cidx].set_xlim(xmin[idx], xmax[idx])
            ax[ridx,cidx].set_ylim(ymin[idx], ymax[idx])     
            
            # ax[ridx,cidx].set_xticks(np.arange(xmin[idx], xmax[idx]+xint[idx], xint[idx]))
            # xtick = np.arange(xmin, xmax+xint, xint*4).tolist()
            # ytick = np.arange(min_alpha, max_alpha+alpha_int, alpha_int*4).tolist()

            # xticks, yticks
            # ax[ridx,cidx].set_xticks([round(xtick[i],2) for i in range(len(xtick))]) 
            # ax[ridx,cidx].set_yticks([round(ytick[i],2) for i in range(len(ytick))])
            
            # replace xtick string 
            # xtick_list = ax[ridx,cidx].get_xticks().tolist()
            # ax[ridx,cidx].set_xticklabels(xtick_list)
            # ytick_list = ax[ridx,cidx].get_yticks().tolist()
            # ax[ridx,cidx].set_yticklabels(ytick_list)
            
            # number of minor ticks
            # ax[ridx,cidx].xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
            # ax[ridx,cidx].yaxis.set_minor_locator(ticker.AutoMinorLocator(4))

            # grid
            ax[ridx,cidx].grid(True, axis='both', linestyle=':', linewidth=0.25, color='0.25', alpha = 0.2); 

            # annotation
            # subplot_idx = '('+subplot_no[idx]+') ' + case_name[idx] #(a) + Internal
            #ax[ridx,cidx].annotate(subplot_idx, xy=(.01, 1.03), xycoords='axes fraction',
                #horizontalalignment='left', verticalalignment='top', fontsize=fs) 
            
            # scale
            ax[ridx,cidx].set_xscale('log')
            ax[ridx,cidx].set_yscale('log')
            
            # legend 
            handles, labels = ax[ridx,cidx].get_legend_handles_labels()
            legorder1 = range(len(handles))
            ax[ridx,cidx].legend([handles[idx] for idx in legorder1],
                                [labels[idx] for idx in legorder1], 
                                loc= "upper left", ncol=1, frameon=False, 
                                edgecolor='None', facecolor='None',
                                fontsize=fs, fancybox=False, 
                                columnspacing= 1.05, labelspacing=0.5,
                                handlelength = 2.5)

            # spine line width  
            for k in ['top','bottom','left','right']:
                    ax[ridx,cidx].spines[k].set_visible(True)
                    ax[ridx,cidx].spines[k].set_linewidth(0.5)
                    ax[ridx,cidx].spines[k].set_color('k')         

            # Addtinal space for colorbar
            fig.subplots_adjust(right=0.85, bottom=0.12) # make space for additional colorbar

