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
from scipy.ndimage.filters import gaussian_filter
warnings.filterwarnings("ignore")
def cm2in(cm):
    return cm/2.54

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

# read the csv file
Z = pd.read_csv("PercentageError.csv") # read the csv file
Z = gaussian_filter(Z, sigma=1)

# figure settings
min_L = 2
max_L = 50
L_int = 2

min_alpha = 0.2
max_alpha = 5
alpha_int = 0.2

norm1 = mcolors.Normalize(vmin = 0, vmax = 90)

# figure settings
nrow = 1
ncol = 1
nfigs = nrow*ncol

x = np.linspace(min_L,max_L,Z.shape[0])
y = np.linspace(min_alpha,max_alpha,Z.shape[1])

# 2D interpolation
f = interp2d(x, y, Z, kind='cubic')

# x, y grid 
x1 = np.linspace(x.min(), x.max(), 75)
y1 = np.linspace(y.min(), y.max(), 75)

X1,Y1 = np.meshgrid(x1, y1)

Z1 =f(x1,y1)


# figure settings
fig, ax = plt.subplots(nrow, ncol, sharex=False, sharey=False, figsize=(cm2in(19),cm2in(15)), 
                        facecolor='w', edgecolor='k', squeeze=False, dpi = 300)

xmin = [2]*nfigs
xmax = [50]*nfigs
xint = [L_int*5]*nfigs
xmar = [0]*nfigs

ymin = [0.2]*nfigs
ymax = [5]*nfigs 
yint = [alpha_int]*nfigs
ymar = [0]*nfigs # yint[i]/6

# figure settings 
for ridx in range(nrow):
    for cidx in range(ncol):
        idx = ridx*ncol + cidx

        ef1 = ax[ridx,cidx].imshow(Z1, aspect='auto', extent=[1,51,5.1,0.1], 
                            cmap='Blues9', norm=norm1, alpha=1.0, interpolation = "spline16") # Error field  interpolation = "spline16" 
        
        ax[ridx,cidx].set_xlabel('System length [cm]', fontsize=FS_label,labelpad=10)
        ax[ridx,cidx].set_ylabel('Thermal diffusivity [10$^{-6}$ m$^2$/s]', fontsize=FS_label,labelpad=10)

        ax[ridx,cidx].tick_params(direction='in', labelsize=Axis_FS, which='major', length=2.5, width=0.5 ,  right=True,top=True ,pad=6)
        ax[ridx,cidx].tick_params(direction='in', labelsize=Axis_FS, which='minor', length=1.25, width=0.25, right=True,top=True ,pad=6)

        # xlim ylim 
        ax[ridx,cidx].set_xlim(xmin[idx]-xmar[idx], xmax[idx]+xmar[idx])
        ax[ridx,cidx].set_ylim(ymin[idx]-ymar[idx], ymax[idx]+ymar[idx])     
        
        # ax[ridx,cidx].set_xticks(np.arange(xmin[idx], xmax[idx]+xint[idx], xint[idx]))
        xtick = np.arange(min_L, max_L+L_int, L_int*4).tolist()
        ytick = np.arange(min_alpha, max_alpha+alpha_int, alpha_int*4).tolist()

        # xticks, yticks
        ax[ridx,cidx].set_xticks([round(xtick[i],2) for i in range(len(xtick))]) 
        ax[ridx,cidx].set_yticks([round(ytick[i],2) for i in range(len(ytick))])
        
        # replace xtick string 
        xtick_list = ax[ridx,cidx].get_xticks().tolist()
        ax[ridx,cidx].set_xticklabels(xtick_list)
        ytick_list = ax[ridx,cidx].get_yticks().tolist()
        ax[ridx,cidx].set_yticklabels(ytick_list)
        
        # number of minor ticks
        ax[ridx,cidx].xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
        ax[ridx,cidx].yaxis.set_minor_locator(ticker.AutoMinorLocator(4))

        # grid
        ax[ridx,cidx].grid(True, axis='both', linestyle=':', linewidth=0.25, color='0.25', alpha = 0.2); 
        # subplot
        # subplot_idx = '('+subplot_no[idx]+') ' + case_name[idx] #(a) + Internal
        #ax[ridx,cidx].annotate(subplot_idx, xy=(.01, 1.03), xycoords='axes fraction',
            #horizontalalignment='left', verticalalignment='top', fontsize=fs) 
                 
        # spine line width  
        for k in ['top','bottom','left','right']:
                ax[ridx,cidx].spines[k].set_visible(True)
                ax[ridx,cidx].spines[k].set_linewidth(0.5)
                ax[ridx,cidx].spines[k].set_color('k')         

# 등고선 그리기
contour_levels = [int(5)]+ [int(10*(i+1)) for i in range(8)]
contour = plt.contour(X1, Y1, Z1, levels=contour_levels, colors='0.1', linewidths=0.75,)
plt.clabel(contour, fontsize=fs-2, fmt = '%1.0f')
plt.tight_layout(pad = 1.7)
fig.subplots_adjust(right=0.85, bottom=0.12) # make space for additional colorbar

# colorbar
cbar_width =  0.015 #vertical  
cbar_height = 0.015 #horizontal  
cbar_dist_v = 0.085; # vertical colorbar distance from bbox edge
cbar_dist_h = 0.028; # horizontal colorbar distance from bbox edge
# blarblar
asdf = "cnrkgoTe"
# horizontal colorbars 
# colorbar1 temperature 
bbox1 = ax[0,0].get_position() # get the normalized axis original position 
cb_ax1 = fig.add_axes([bbox1.x1+cbar_dist_h, bbox1.y0, cbar_width, bbox1.y1-bbox1.y0]) #[x_origin, y_origin, width, height]
cbar1  = fig.colorbar(ef1, cax=cb_ax1, ax=ax[0,0], orientation='vertical') 
cbar1.ax.tick_params(direction='in',labelsize=Axis_FS, length=2, width=0.5, pad=3)
cbar1.ax.minorticks_off()
cbar1.locator = ticker.MultipleLocator(10)
cbar1.ax.set_ylabel('Percentage error [%]', rotation=90, fontsize=FS_label, labelpad=10)
cbar1.outline.set_linewidth(0.5)

figname = 'Fig2 Percentage error Comparison'
# plt.savefig(figname+'.svg', dpi=600)
# plt.savefig(figname+'.png', dpi=600)
# plt.savefig(figname+'.pdf', dpi=600)
