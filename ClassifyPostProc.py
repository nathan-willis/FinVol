import numpy as np
import matplotlib.pyplot as plt
import sys
if float(str(sys.version_info[1]) + '.' + str(sys.version_info[2])) <= 9.1:
    from matplotlib import cm
else:
    from matplotlib import colormaps as cm
import matplotlib.animation as ani
import matplotlib.ticker as tkr
from matplotlib.patches import Rectangle, FancyArrowPatch
from celluloid import Camera
import os
import seaborn as sns
from copy import deepcopy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from scipy.ndimage import gaussian_filter1d

"""
load file from finite volume solver
the first row should be the initial time and then spatial vector of the cell averages, the time can be thrown away

Every row after the first should be the time followed by the cell averages
"""

def article_params():
    plt.rcParams.update(
        {
            "text.usetex":True,
            "figure.figsize":[3.5,3],
            "font.family":"serif",
            "font.sans-serif":["Helvetica"],
            'font.size':10,
            'lines.linewidth':2,
            'legend.fontsize':9,
            'xtick.labelsize':8,
            'ytick.labelsize':8
        }
    )
def viewing_params():
    plt.rcParams.update(
        {
            "text.usetex":False,
            "figure.figsize":[12,9],
            "font.family":"serif",
            "font.sans-serif":["Helvetica"],
            'font.size':22,
            'lines.linewidth':3,
            'legend.fontsize':20,
            'xtick.labelsize':20,
            'ytick.labelsize':20
        }
    )
def panel_label(ax,subplot_number = None):
    label_list = ['a','b','c','d','e','f','g','h']
    label = f"({label_list[subplot_number if subplot_number else ax.get_subplotspec().num1]})"
    X = ax.get_xlim()
    Y = ax.get_ylim()
    x_pos = X[0] + 0.04*(X[1]-X[0])
    y_pos = Y[0] + 0.80*(Y[1]-Y[0])
    plt.text(
        x_pos,
        y_pos,
        label,
        horizontalalignment = 'center',
        verticalalignment = 'center',
        bbox=dict(facecolor='white',alpha = 0.75,linewidth=0,boxstyle='round,pad=0.3')
    )
viewing_params()
tabcolors = ('tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan')
mpllinestyles = ('solid','dashed','dotted','dashdot',(0, (3, 5, 1, 5, 1, 5)))
mpllinestyles = ('solid','dashed','dotted','dashdot')
mplmarkers = ('o','v','^','<','>','s','*','D','P','X')

variable_dict = {'c':'concentration', 'c1':'left concentration', 'c2':'right concentration', 'u':'velocity', 'q':'q, conserved velocity', 'phi':'phi, conserved concentration', 'h':'height','h_latex':'$h(x,t)$','u_latex':'$u(x,t)$','c1_latex':'$c_1(x,t)$','c2_latex':'$c_2(x,t)$','d1':'deposit 1','d2':'deposit 2'}
char_to_cons = {'u':'q','c1':'phi1','c2':'phi2'}
char_vars = ['u','c1','c2']

testFileName = 'SedimentationInitialConditionTest_2025Jun7/'

def load_settling_sims(US=[0,0.005,0.01,0.015,0.02]):
    S = {}
    for us in US:
        S[us] = TurbiditySim(1.,1.,us,'Nov24_AsymBoxModel/',['h','u','c1','c2'],sharp = 100,N=14000)
    return S

def intersection(x,f,g):
    '''
    Find the intersection points between two functions (as vectors) f and g
    The nodes (x), must be provided as well. 
    '''
    y = f-g
    sign_changes = list(np.argwhere(y[:-1]*y[1:]<=0).flatten())
    roots = []
    for sc in sign_changes:
        x0,x1,y0,y1 = x[sc],x[sc+1],y[sc],y[sc+1]
        roots.append(x1-y1*(x0-x1)/(y0-y1))
    return (None if len(roots) == 0 else roots[0]) if len(roots)<=1 else roots

class TurbiditySim:

    def __init__(self, hR0, cR0, U_s, rootFile,VARS, N = 5000, NuRe = 1000, finalTime = 40., h_min = 0.0001, CFL = 0.1, sharp = 50, apart = 5., FrSquared = 1., hL0 = 1.0, cL0 = 1.0, NuPe = None, subFile = 'sims/'):
        self.hR0 = hR0
        self.cR0 = cR0
        self.U_s = U_s
        self.rootFile = rootFile + '' if rootFile[-1] == '/' else rootFile + '/'
        self.subFile = subFile
        self.N = N
        self.NuRe = NuRe
        if NuPe == None: self.NuPe = NuRe
        self.finalTime = finalTime
        self.h_min = h_min
        self.CFL = CFL
        self.sharp = sharp
        self.apart = apart
        self.FrSquared = FrSquared
        self.hL0 = hL0
        self.cL0 = cL0

        self.fileName = "hTwo%0.2f_cTwo%0.2f_%iapart_N%i_CFL%0.3f_T%0.1f_NuRe%i_Us%0.3f_hmin%0.5f_sharp%i"%(self.hR0,self.cR0,self.apart,self.N,self.CFL,self.finalTime,self.NuRe,self.U_s,self.h_min,self.sharp)
        try:
            with open(self.rootFile + self.subFile + self.fileName + '/info.log'):
                pass
        except FileNotFoundError as err:
            print('FileNotFoundError:',err)
            self.fileName = "hOne%0.2f_hTwo%0.2f_cOne%0.2f_cTwo%0.2f_%iapart_N%i_CFL%0.3f_T%0.1f_NuRe%i_NuPe%i_FrFr%0.3f_Us%0.3f_hmin%0.5f_sharp%i"%(self.hL0,self.hR0,self.cL0,self.cR0,self.apart,self.N,self.CFL,self.finalTime,self.NuRe,self.NuPe,self.FrSquared,self.U_s,self.h_min,self.sharp)

        self.unpack(VARS)
        try:
            self.dt = self.T[1]-self.T[0]
        except IndexError:
            self.dt = np.nan
        self.dx = self.x[1]-self.x[0]
        with open(self.rootFile + self.subFile + self.fileName + '/info.log') as fh:
            for line in fh:
                if 'Run time was ' in line:
                    self.runTime = float(line[line.find('was')+4:line.find('seconds')-1])
                if 'collision time:' in line:
                    self.coll_time = float(line[line.find('=')+1:])
                if 'collision index:' in line:
                    self.coll_idx = int(float(line[line.find('=')+1:]))
                if 'collision position:' in line:
                    self.coll_loc = float(line[line.find('=')+1:])

    def unpack(self, whichVars):
        def single_unpack(whichFile):
            """
            Argument is the file name of the data
            When loading data, Each row is a ''time slice'' which is a vector of the function values at each point in space

            The first row is the initial time and then spatial vector of the center of the cells
            The first column is the time values, with the initial time listed twice, see above comment
            Removing the first column and first row you are left with a matrix of the function values (rows are time slices)
            """
            if whichFile in char_vars:
                A = np.loadtxt(self.rootFile + self.subFile + self.fileName + '/' + char_to_cons[whichFile])
            else:
                A = np.loadtxt(self.rootFile + self.subFile + self.fileName + '/' + whichFile)
            self.x = A[0,1:]
            self.T = A[1:,0]
            DependentVariable = A[1:,1:]
            if whichFile in char_vars:
                try:
                    DependentVariable = DependentVariable/self.h
                except AttributeError:
                    self.h = single_unpack('h')
                    DependentVariable = DependentVariable/self.h
            return DependentVariable

        for var in whichVars:
            setattr(self,var,single_unpack(var))

    def sim_info(self):
            # print the log file to screen. 
            print('')
            with open(self.rootFile + self.subFile + self.fileName + '/info.log') as fh:
                for line in fh: print(line)

    def makeMP4(self, varList = ['h','u','c1','c2'],tMax=1000.,show_legend = False, xlim = None, ylim = None,framerate = 30.):#xmax = None, xmin = None,ymax = None, ymin = ):
        plt.rcParams.update({"text.usetex":False})
        fig = plt.figure(figsize=(12,8))
        camera = Camera(fig)

        U = []
        for var in varList:
            U.append(getattr(self,var))

        if ylim:
            plt.ylim(ylim)
            ymin1,ymax1 = ylim
            ymin2,ymax2 = ylim
        else:
            ymin1,ymax1 = min(np.min(U[0]),np.min(U[1])),max(np.max(U[0]),np.max(U[1]))
            ymin2,ymax2 = min(np.min(U[2]),np.min(U[3])),max(np.max(U[2]),np.max(U[3]))
        maxVEL = 0.

        for ii in range(len(self.T)):
            if self.T[ii]>tMax: continue
            subplotcounter = 1
            for jj in range(len(varList)):
                subplotcounter += 1
                u = U[jj]
                plt.subplot(2,1,int(subplotcounter/2))
                plot_ = plt.plot(self.x,u[ii,:],color=tabcolors[jj],linestyle = mpllinestyles[0])
                plt.grid()

            plt.subplot(211)
            plt.xlim(xlim)
            plt.ylim([ymin1,ymax1])
            VEL = U[1]
            maxVEL = max(maxVEL,np.max(VEL[ii,:]))
            timeStr = 't = %0.2f, u* = %0.2f, uM = %0.2f, Fr = %0.3f'%(self.T[ii],np.max(VEL[ii,:]), maxVEL,self.FrSquared)
            plt.text((xlim[0]+xlim[1])/2 if xlim else 0., ymax1 + 0.1*(ymax1-ymin1),timeStr,verticalalignment = 'center',horizontalalignment = 'center')

            plt.subplot(212)
            #plt.xlim([xmin,xmax])
            plt.xlim(xlim)
            #plt.ylim([ymin2,ymax2])
            print(self.T[ii])
            camera.snap()

        if show_legend:
            Legend = []
            for var in varList:
                Legend.append(variable_dict[var])
            plt.subplot(211)
            plt.legend(Legend[:2],loc= 'upper left')
            plt.subplot(212)
            plt.legend(Legend[2:],loc= 'upper left')

        animat = camera.animate()

        Writ = ani.FFMpegWriter(fps=framerate, metadata=dict(artist='nathan'))
        VidPath = os.getcwd()[:os.getcwd().find('/D')+1] + "Documents/MercedResearch/WithFrancois/Turbidity/FinVol/" + self.rootFile + "solutions/videos/"
        animat.save(VidPath + self.fileName.replace('.','_') + '_tMax%i'%tMax + '.mp4', writer = Writ)
        plt.close()

    def get_oneSided_boxModel_vertices(self,t,T,xN,xB,hP,hM):
        '''
        Given vectors for time, xB (bore pos.), xN (front/nose pos.)
        hP (h_plus), and hM (h_minus) return the vertices for the 2 lines
        that can draw the one-sided box model.

        I return 2 separate lists to draw two lines, becuase one long list will
        double draw the box in some region, which is fine for soild lines, but
        is noticeable for dashed lines.

        WARNING! t is the original time, but T is shift so that T=0 is coll.
        t-self.coll_time shifts t to be in the same time frame as T.
        '''
        if t<self.coll_time:
            return [[None]*4]*2,[[None]*3]*2
        idx = np.argmin(np.abs(T-(t-self.coll_time)))
        xb,xn = xB[idx],xN[idx]
        hp,hm = hP[idx],hM[idx]
        xc = self.coll_loc

        return [[xc,xn,xn,xb],[0,0,hp,hp]],[[xc,xb,xb],[hm,hm,0]]

    def BoxSWE_MP4(self, tMax=1e8, xlim = [0,None], framerate = 30., shape_factor = 0.9):
        '''
        This function generates an mp4 coomparing shallow-water data to box model.
        There are two subplots in the mp4, the first (top) is an a fixed axis,
        which is everything to the right of the collision point;
        the second (bottom) has dynamic axes so that the current, to the right of the
        collision point, fills the view.

        The no shape factor case (shape_factor=1) is always shown, and a second version
        with a shape_factor (shape_factor = 0.9 is default) is also plotted.
        '''
        article_params()
        plt.rcParams.update({"text.usetex":False})

        fig,ax = plt.subplots(2,1,figsize=(6,4))
        t_num,   xN,   xB,   hP,   hM,   uP,   cM,   cP,   B_v    = self.settling_RH_model(hmf = 1.0)
        t_num_SF,xN_SF,xB_SF,hP_SF,hM_SF,uP_SF,cM_SF,cP_SF,B_v_SF = self.settling_RH_model(hmf = shape_factor) # SF subscript to denote shape factor
        L1,L2       = self.get_oneSided_boxModel_vertices(0,t_num,xN,xB,hP,hM)
        L1_SF,L2_SF = self.get_oneSided_boxModel_vertices(0,t_num_SF,xN_SF,xB_SF,hP_SF,hM_SF)
        # Plot 1
        Plot1_SWline, = ax[0].plot(self.x,self.h[0,:],label = 'shallow water')
        Plot1_BMline1, = ax[0].plot(L1[0],L1[1],linestyle = 'dashed',color ='k',label = 'box model')
        Plot1_BMline2, = ax[0].plot(L2[0],L2[1],linestyle = 'dashed',color ='k')
        Plot1_BMline1_SF, = ax[0].plot(L1_SF[0],L1_SF[1],linestyle = 'dashed',color ='r',label = 'box model, with shape factor')
        Plot1_BMline2_SF, = ax[0].plot(L2_SF[0],L2_SF[1],linestyle = 'dashed',color ='r')
        ax[0].set_xlim(self.coll_loc,self.x[-1])
        ax[0].set_ylim(-0.05,self.h.max()+0.05)
        ax[0].grid()
        ax[0].legend()
        # Plot 2
        Plot2_SWline, = ax[1].plot(self.x,self.h[0,:])
        Plot2_BMline1, = ax[1].plot(L1[0],L1[1],linestyle = 'dashed',color ='k')
        Plot2_BMline2, = ax[1].plot(L2[0],L2[1],linestyle = 'dashed',color ='k')
        Plot2_BMline1_SF, = ax[1].plot(L1_SF[0],L1_SF[1],linestyle = 'dashed',color ='r')
        Plot2_BMline2_SF, = ax[1].plot(L2_SF[0],L2_SF[1],linestyle = 'dashed',color ='r')
        def func(t):
            L1,L2       = self.get_oneSided_boxModel_vertices(t,t_num,xN,xB,hP,hM)
            L1_SF,L2_SF = self.get_oneSided_boxModel_vertices(t,t_num_SF,xN_SF,xB_SF,hP_SF,hM_SF)
            h_SW = self.h[np.argmin(np.abs(self.T-t)),:]
            # Plot 1 update
            Plot1_SWline.set_ydata(h_SW)
            Plot1_BMline1.set_xdata(L1[0])
            Plot1_BMline1.set_ydata(L1[1])
            Plot1_BMline2.set_xdata(L2[0])
            Plot1_BMline2.set_ydata(L2[1])
            Plot1_BMline1_SF.set_xdata(L1_SF[0])
            Plot1_BMline1_SF.set_ydata(L1_SF[1])
            Plot1_BMline2_SF.set_xdata(L2_SF[0])
            Plot1_BMline2_SF.set_ydata(L2_SF[1])
            # Plot 2 update
            Plot2_SWline.set_ydata(h_SW)
            Plot2_BMline1.set_xdata(L1[0])
            Plot2_BMline1.set_ydata(L1[1])
            Plot2_BMline2.set_xdata(L2[0])
            Plot2_BMline2.set_ydata(L2[1])
            Plot2_BMline1_SF.set_xdata(L1_SF[0])
            Plot2_BMline1_SF.set_ydata(L1_SF[1])
            Plot2_BMline2_SF.set_xdata(L2_SF[0])
            Plot2_BMline2_SF.set_ydata(L2_SF[1])

            ax[1].set_xlim(self.coll_loc,max(5,self.x[np.argwhere(h_SW>2*self.h_min)[-1][0]]+0.2))
            ax[1].set_ylim(h_SW.min(),h_SW.max())

            ax[0].set_title('t = %0.2f'%(t))
            #plt.text((xlim[0]+xlim[1])/2 if xlim else 0., ymax + 0.1*(ymax-ymin),timeStr,verticalalignment = 'center',horizontalalignment = 'center')

            print('t = %0.2f'%(t))
            return Plot1_SWline, Plot2_SWline, Plot1_SWline, Plot1_BMline1, Plot1_BMline2, Plot1_BMline1_SF, Plot2_BMline1, Plot2_BMline2, Plot2_BMline1_SF, Plot2_BMline2_SF


        anim = ani.FuncAnimation(fig,func,frames = self.T[self.T<tMax],blit=False)
        Writ = ani.FFMpegWriter(fps=framerate, metadata=dict(artist='nathan'))
        VidPath = os.getcwd()[:os.getcwd().find('/D')+1] + "Documents/MercedResearch/WithFrancois/Turbidity/FinVol/" + self.rootFile + "solutions/videos/BoxSWE_shapefactor%0.1f_"%shape_factor
        tMaxString = '_tMax%i'%tMax if tMax<100 else ''
        anim.save(VidPath + self.fileName.replace('.','_') + tMaxString + '.mp4', writer = Writ)
        plt.close()

    def plot_height_conc_time(self,desired_time,xlim = None,cb=True,cb_choice='initial',cm='BrBG'):
        index = np.argmin(np.abs(self.T-desired_time))
        x_min_idx, x_max_idx = (np.argmin(np.abs(self.x-xlim[0])), np.argmin(np.abs(self.x-xlim[1]))) if xlim else (None, None)
        x_ = self.x[x_min_idx:x_max_idx]
        try:
            h_ = self.h[index,x_min_idx:x_max_idx]
        except AttributeError:
            self.unpack(['h'])
            h_ = self.h[index,x_min_idx:x_max_idx]
        try:
            c1_ = self.c1[index,x_min_idx:x_max_idx]
        except AttributeError:
            self.unpack(['c1'])
            c1_ = self.c1[index,x_min_idx:x_max_idx]
        try:
            c2_ = self.c2[index,x_min_idx:x_max_idx]
        except AttributeError:
            self.unpack(['c2'])
            c2_ = self.c2[index,x_min_idx:x_max_idx]


        polygon = plt.fill_between(x_,h_*0,h_,lw=1,color='none')
        #gradient_denominator = np.array([c1_[ii]+c2_[ii] if c1_[ii]+c2_[ii]!=0 else 1 for ii in range(len(self.x))])
        verts = np.vstack([p.vertices for p in polygon.get_paths()])
        gradient = plt.imshow(np.reshape(c1_-c2_,(1,-1)),
                              cmap = cm,
                              vmax=self.cL0 if cb_choice=='initial' else None,vmin = -self.cR0 if cb_choice=='initial' else None,
                              aspect = 'auto',
                              extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()]
                             )
        gradient.set_clip_path(polygon.get_paths()[0], transform=plt.gca().transData)
        #plt.plot(self.x,c1_+c2_,color='k',linewidth=1)
        plt.plot(x_,h_)

        if cb:
            cbar=plt.colorbar(gradient)
            if cb_choice == 'initial':
                cbar.set_ticks(ticks=[self.cL0,-self.cR0])
                cbar.ax.set_yticklabels(['$100\%\ c_2$','$100\%\ c_1$'])
            elif cb_choice == 'dynamic':
                cbar.set_ticks(ticks=[np.min(c1_-c2_),np.max(c1_-c2_)])
                cbar.ax.set_yticklabels(['$100\%\ c_2$','$100\%\ c_1$'])

        plt.xlabel('$x$')
        plt.ylabel(variable_dict['h'])

    def plot_time(self,var,desired_time,xlim = None,linestyle=None,color=None):
        index = np.argmin(np.abs(self.T-desired_time))
        x_min_idx, x_max_idx = (np.argmin(np.abs(self.x-xlim[0])), np.argmin(np.abs(self.x-xlim[1]))) if xlim else (None, None)

        plt.plot(self.x[x_min_idx:x_max_idx],getattr(self,var)[index,x_min_idx:x_max_idx],label = '$t=%0.1f$'%self.T[index],linestyle=linestyle,color=color)

        plt.xlabel('$x$')
        plt.ylabel(variable_dict[var+'_latex'])
        plt.xlim(xlim)

    def plot_times(self,var,times,xlim=None,wl = True,linestyle=None,color=None):
        for t in times:
            self.plot_time(var,t,color=color,linestyle=linestyle)
        plt.xlim(xlim)
        if wl: plt.legend()


    def plot_profile_results(self,var,times=[0,4,8,12]):
        article_params()
        plt.figure(figsize = [5.125,4])
        for i,v in enumerate(var):
            plt.subplot(len(var),1,i+1)
            self.plot_times(v,times,xlim=[-25,25],wl=False)
            plt.ylabel(variable_dict[v + '_latex'])
            # if i < len(var)-1: plt.gca().set_xticks([])
            if i < len(var)-1: plt.gca().tick_params(axis='x', labelbottom=False)
        # Make the vetical axis bounds be symmetric across 0 for velocity. 
        if 'u' in var:
            plt.subplot(len(var),1,var.index('u')+1)
            vel_bound = np.max(np.abs(plt.gca().get_ylim()))
            plt.ylim([-vel_bound,vel_bound])
        # Make the concentrations plot with the same vertical axis 
        # Maybe these should be fixed at 1, currently I find the max ymax and set that for both.
        if 'c1' in var and 'c2' in var:
            # Fine which concentration has higher max
            plt.subplot(len(var),1,var.index('c1')+1)
            cmax1 = plt.gca().get_ylim()[1]
            plt.subplot(len(var),1,var.index('c2')+1)
            cmax2 = plt.gca().get_ylim()[1]
            cmax  = max(cmax1,cmax2)
            # Set that max value for both plots. 
            plt.subplot(len(var),1,var.index('c1')+1)
            plt.ylim([None,cmax])
            plt.subplot(len(var),1,var.index('c2')+1)
            plt.ylim([None,cmax])
        # Set legend of times above top plot
        plt.subplot(len(var),1,1)
        plt.legend(
            ['$t=%0.1f$'%t for t in times],
            ncol=len(times),
            loc='upper center',
            bbox_to_anchor=(0.5, 1.4)
        )
        # Set axis for bottom plot
        plt.subplot(len(var),1,len(var))
        plt.xlabel('$x$')
        # Add panels 
        for i in range(1,len(var)+1):
            plt.subplot(len(var),1,i)
            panel_label(plt.gca())

        # Format and save
        plt.subplots_adjust(right=0.99,left =0.1,bottom=0.1,top =0.925,hspace=0.1)
        plt.savefig(
            self.rootFile + 'solutions/plots/solution_example_' + self.fileName + '.png',
            bbox_inches='tight',
            dpi=400
        )
        plt.savefig(
            self.rootFile + 'solutions/plots/solution_example_' + self.fileName + '.pdf',
            bbox_inches='tight',
        )

    def deposition_details(self):
        dx = self.x[1]-self.x[0]

        self.xr = self.x[self.coll_idx+1:int(self.N*0.8)]
        self.l2r = self.d1[-1,self.coll_idx+1:int(self.N*0.8)]

        self.xl = self.x[int(self.N*0.2):self.coll_idx+1]
        self.r2l = self.d2[-1,int(self.N*0.2):self.coll_idx+1]

        two_sided_encroachment = np.hstack([self.r2l,self.l2r])
        two_sided_x = np.hstack([self.xl,self.xr])

        l2r_mass = np.sum(self.l2r)*dx
        r2l_mass = np.sum(self.r2l)*dx

        #self.encroachment_mass = l2r_mass if l2r_mass >= r2l_mass else r2l_mass
        #self.COM_x = np.sum(self.xr*self.l2r)*dx/l2r_mass if l2r_mass >= r2l_mass else np.sum(self.xl*self.r2l)*dx/r2l_mass
        self.encroachment_mass = l2r_mass - r2l_mass
        self.COM_x = np.sum(two_sided_x*two_sided_encroachment)*dx/(l2r_mass + r2l_mass)

    def plot_deposit(self, LS = 'solid'):
        plt.plot(self.x,self.d1[-1,:],label='$d_1(x), U_s = %0.3f$'%self.U_s,color = 'tab:blue', linestyle = LS)
        plt.plot(self.x,self.d2[-1,:],label='$d_2(x), U_s = %0.3f$'%self.U_s,color = 'tab:orange', linestyle = LS)
        plt.xlim([-20,20])
        plt.title('$h_{2,0}$ = %0.2f, $c_{2,0}$ = %0.2f'%(self.hR0,self.cR0))
        plt.xlabel('$x$')

    def plot_deposit_gradient(self,dt_plot=1,xb_tol=1e-5,xb=None,yb=None,show=True,save = False, close = True,cb = True,title=True):
        article_params()
        plt.figure(figsize=[5.125,1.2 + 0.2*int(title)])
        for desired_time in np.flip(self.T[self.T%dt_plot<0.9*self.dt])[:-1]:
            print(desired_time)
            tI = np.argmin(np.abs(self.T-desired_time)) # tI for time index.
            polygon = plt.fill_between(self.x,self.d1[tI,:]*0,self.d1[tI,:]+self.d2[tI,:],lw=1,color='none')
            gradient_denominator = np.array([self.d1[tI,ii]+self.d2[tI,ii] if self.d1[tI,ii]+self.d2[tI,ii]!=0 else 1 for ii in range(len(self.x))])
            verts = np.vstack([p.vertices for p in polygon.get_paths()])
            gradient = plt.imshow(np.reshape(self.d1[tI,:]/gradient_denominator,(1,-1)),cmap = 'PRGn',aspect = 'auto',extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
            gradient.set_clip_path(polygon.get_paths()[0], transform=plt.gca().transData)
        plt.plot(self.x,self.d1[-1,:]+self.d2[-2,:],color='k',linewidth=1)

        if cb:
            cbar=plt.colorbar(gradient,aspect=5,pad=0.01)
            cbar.set_ticks(ticks=[0,1])
            cbar.ax.set_yticklabels(['$100\%\ d_2$','$100\%\ d_1$'])

        plt.xlabel('$x$',labelpad=0)
        #plt.ylabel('$d_{f,1}+d_{f,2}$')
        plt.ylabel('deposit height')

        xb = xb if xb else max(np.abs(self.x[self.d1[-1,:]>xb_tol][0]),np.abs(self.x[self.d2[-1,:]>xb_tol][-1]))
        plt.xlim([-xb,xb])
        plt.ylim([0,yb])
        if title:
            plt.title('$h_r=%0.2f$, $c_r=%0.2f$'%(self.hR0,self.cR0))
            plt.subplots_adjust(left = 0.1, right =0.99,bottom=0.23,top =0.84)
        else:
            plt.subplots_adjust(left = 0.1, right =0.99,bottom=0.3,top =0.91)
        if show: plt.show()
        if save: plt.savefig(self.rootFile + 'solutions/plots/gradientDeposit_' + self.fileName + '.png',dpi=1000)
        if save: plt.savefig(self.rootFile + 'solutions/plots/gradientDeposit_' + self.fileName + '.pdf',dpi=1000)
        if close: plt.close()

    def plot_encroachment(self,wantLegend=True, want_y_label=True):
        try:
            self.encroachment_mass
        except AttributeError:
            self.deposition_details()
        plt.plot(self.xl,self.r2l,label = 'right-to-left: $d_2(x), x>x_c$')
        plt.plot(self.xr,self.l2r,label = 'left-to-right: $d_1(x), x<x_c$')
        #plt.scatter(self.COM_x,self.encroachment_mass,)
        plt.plot([self.coll_loc]*2,[0,max(np.max(self.l2r),np.max(self.r2l))],color = 'k',linestyle = 'dashed',label = 'collision location $(x_c)$: %0.2f'%self.coll_loc,linewidth = 2)
        plt.plot([self.COM_x]*2,[0,max(np.max(self.l2r),np.max(self.r2l))],color = 'red',linestyle = 'dashed',label = 'COM $x$-coordinate: %0.2f'%self.COM_x)
        try:
            xMin = self.xl[np.argwhere(self.r2l>0.01*np.max(self.r2l))[0][0]]
            xMax = self.xr[np.argwhere(self.l2r>0.01*np.max(self.l2r))[-1][0]]
            plt.xlim([xMin,xMax])
        except IndexError:
            print('Adaptive xlim did not work, setting bounds at [-5,5]')
            plt.xlim([-5,5])
        plt.title('$h_{2,0}$ = %0.2f, $c_{2,0}$ = %0.2f \n $\int_{x_c}^\infty d_1(x) dx - \int_{-\infty}^{x_c} d_2(x) dx $= %0.2f'%(self.hR0,self.cR0,self.encroachment_mass))
        if wantLegend: plt.legend()
        plt.xlabel('$x$')
        if want_y_label: plt.ylabel('deposits \n left: $d_1(x)$, right: $d_2(x)$')

    def sus_conc(self,LS = None, LC = None):
        c = (self.c1 + self.c2)*self.h
        dx = self.x[1]-self.x[0]
        initial = np.sum(c[0,1:] + c[0,:-1])*(dx/2)
        suspended_pct = np.sum(c[:,1:] + c[:,:-1],1)*(dx/2)/initial
        plt.plot(self.T,suspended_pct, color = LC, linestyle = LS)
        return self.T[np.argwhere(suspended_pct > 0.05)[-1][0]]

    def bore_data(self,subSampleBy=1,all_var=True,t_start=0,plot=False,save = True):
        def average_cp_cm(self,xB,CR_temp,front):
            def avg(f,a,b):
                f_int = (self.dx/2)*np.sum(f[:-1]+f[1:])
                return f_int/(b-a)
            b_idx = np.argmin(np.abs(self.x-xB))
            a_bd = self.x[CR_temp>0.1*CR_temp.max()][0]
            cp = CR_temp[b_idx:]
            cm = CR_temp[:b_idx]
            return avg(cm,a_bd,xB),avg(cp,xB,front)
        try:
            C=np.loadtxt(self.rootFile + 'solutions/postData/post_collision_bore_data_'+ self.fileName + '.csv',delimiter=',')
            t_post = C[:,0]
            bore = C[:,1]
            h_plus,   h_minus   = C[:,2], C[:,3]
            u_plus,   u_minus   = C[:,4], C[:,5]
            h_plus_avg,   h_minus_avg   = C[:,6], C[:,7]
            c_plus_avg,   c_minus_avg   = C[:,8], C[:,9]
            front = C[:,10]
        except OSError:
            print('cannot open, so I will post process the data MYSELF!')
            t_post = []
            u_plus, u_minus = [],[]
            h_plus, h_minus = [],[]
            h_plus_avg, h_minus_avg = [],[]
            c_plus_avg, c_minus_avg = [],[]
            bore, front = [],[]

            dt = self.T[1]-self.T[0]
            post_collision = False # Flag to determine if the currents have collided yet 
            h_max = 0 # Intialize h_max to be 0, so that at the first iteration it will be smaller than actual h_max and replaced with h_max
            bore_index = self.coll_idx # Initialize the bore index as the collision index

            for t,u_fake,h_fake,c_fake in zip(self.T,self.u,self.h,self.c2):
                u_ = deepcopy(u_fake) # It appeared that within this loop the data was getting overwritten with u_ (which does not make sense), the deepcopy forces this to NOT happen
                h_ = deepcopy(h_fake) # Same as above
                c_ = deepcopy(c_fake) # Same as above
                window_size = int(self.N*0.0025) # Create a "viewing window" so that the search for the bore is only local to the previous bore. 
                if (not post_collision) and h_[bore_index]>2*self.h_min and np.max(h_)<h_max and self.x[np.argmax(h_)]<2 and t>max(t_start,self.coll_time):
                    '''
                    There are four conditions (I don't really count t>t_start) we want before tracking the bore.
                      1) not post_collision is there to make sure we never flip this flag more than once.
                      2) h_[bore]>2*h_min checks that the currents have reach the collision point.
                      3) After collsion, the fluids jet "up" and this region is fairly noisy, we will wait until the jet starts to recede.
                         The maximum will be within the jet, we donlt want to track the bore until the current max is smaller than the previous max.
                         h_max is updated below.
                      4) Right after collision, the argmax of h should be near the bore, which should be somewhat near the origin. This is mainly to avoid to "snowplow" being the max.

                      **5) Sometimes I want to force it to start later to ignore and difficulties immediately following collision
                    '''
                    post_collision = True

                h_max = np.max(h_)
                front_index = np.argwhere(h_>2*self.h_min)[-1][0] # Right traveling front.

                if post_collision:
                    # threshold = (h_[int((front_index+bore_index)/2)] + h_[int((self.coll_idx+bore_index)/2)])/2
                    # u_max = np.max(u_)
                    # if bore_index:
                    #      h_[:max(bore_index-2*window_size,0)]=np.max(h_) # Set everything "sufficiently far" behind the bore to be h_max
                    #      h_[min(bore_index+2*window_size,self.N):]=self.h_min # Set everythign "sufficiently far" ahead of the bore to be h_min
                    # breakpoint()
                    # bore_index = np.argwhere(h_>threshold)[-1][0]
                    bore_local_search = range(max(bore_index-window_size,0),min(bore_index+window_size,self.N)+1) # Only search locally around the bore
                    u_bore = u_[bore_local_search]
                    h_bore = h_[bore_local_search]
                    x_bore = self.x[bore_local_search]
                    threshold = (np.max(h_bore)+np.min(h_bore))/2
                    bore_index_local = np.argwhere(h_bore>threshold).reshape(-1)[-1]
                    bore_index = np.argmin(np.abs(self.x - x_bore[bore_index_local]))

                    u_mbi, u_pbi = np.argmax(u_bore), np.argmin(u_bore) # u_mbi is u_minus bore index, u_pbi is u_plus bore index

                    bore_loc = (self.x[bore_index]-self.x[bore_index+1])*(threshold-h_[bore_index+1])/(h_[bore_index]-h_[bore_index+1])+self.x[bore_index+1] # Linear approximation between nodes values for height immediately above/below threshold
                    if bore_loc < 0.2: continue
                    h_minus_temp, h_plus_temp = h_bore[u_mbi], h_bore[u_pbi]
                    new_thresh = (h_minus_temp+h_plus_temp)/2
                    #bore_local_index = np.argwhere(h_bore>new_thresh)[-1][0]

                    #bore_local_loc = (x_bore[bore_local_index]-x_bore[bore_local_index+1])*(new_thresh-h_bore[bore_local_index+1])/(h_bore[bore_local_index]-h_bore[bore_local_index+1])+x_bore[bore_local_index+1] # Linear approximation between nodes values for height immediately above/below threshold
                    #print(bore_local_loc,bore_loc)

                    t_post.append(t-self.coll_time)
                    u_minus.append(u_bore[u_mbi])
                    u_plus.append(u_bore[u_pbi])
                    h_minus.append(h_bore[u_mbi])
                    h_plus.append(h_bore[u_pbi])

                    front_loc = self.x[front_index]

                    cmA, cpA = average_cp_cm(self,bore_loc,c_,front_loc)
                    print(t,cpA,cmA)

                    HMA = np.sum(h_fake[self.coll_idx:bore_index])*self.dx/(bore_loc-self.coll_loc)
                    HPA = np.sum(h_fake[bore_index:front_index])*self.dx/(front_loc-bore_loc)
                    h_minus_avg.append(HMA)
                    h_plus_avg.append(HPA)
                    c_minus_avg.append(cmA)
                    c_plus_avg.append(cpA)
                    bore.append(bore_loc)
                    #front_loc = (x[front_index]-x[front_index+1])*(2*h_min-h_[front_index+1])/(h_[front_index]-h_[front_index+1])+x[front_index+1]
                    front.append(front_loc)
                if plot:
                    plt.subplot(212)
                    plt.xlabel('x')
                    plt.ylabel('velocity')

                    plt.subplot(211)
                    plt.xlabel('x')
                    plt.ylabel('height')

                    plt.legend()

            array_to_save=np.array((np.array(t_post),np.array(bore),np.array(h_plus),np.array(h_minus),np.array(u_plus),np.array(u_minus),np.array(h_plus_avg),np.array(h_minus_avg),np.array(c_plus_avg),np.array(c_minus_avg),np.array(front))).T
            if save: np.savetxt(self.rootFile + 'solutions/postData/post_collision_bore_data_' + self.fileName + '.csv',array_to_save,delimiter=',')
        def subsample(x,subSampleBy = subSampleBy):
            x = np.array(x)
            return x[range(0,x.shape[0],subSampleBy)]
        self.t_post = subsample(t_post)
        self.bore = subsample(bore)
        self.hP_data, self.hM_data = subsample(h_plus),subsample(h_minus)
        self.hP_data_avg, self.hM_data_avg = subsample(h_plus_avg),subsample(h_minus_avg)
        self.cP_data_avg, self.cM_data_avg = subsample(c_plus_avg),subsample(c_minus_avg)
        self.uP_data, self.uM_data = subsample(u_plus),subsample(u_minus)
        self.front_data = subsample(front)

    def settling_RH_model(self, t0=0, dt=0.001, final=np.inf, hmf=1, tol=1e-14):
        def get_collision_concentration():
            collision_time_index = np.argmin(np.abs(self.T-self.coll_time))
            CR_temp = self.c2[collision_time_index,:]
            CR_area = (1/2)*np.sum(CR_temp[1:]+CR_temp[:-1])*self.dx
            CR_bounds = self.x[CR_temp>0.1][[0,-1]]
            return CR_area/(CR_bounds[1]-CR_bounds[0])
        self.front_vel()
        xC = self.coll_loc # xC is the collision point. 
        uN = self.uN
        xI = self.apart/2.
        VR  = 1.*self.hR0
        cR = self.cR0
        cC = get_collision_concentration()
        us = self.U_s

        print('\n' + '*'*43 + '\n* t0  = %0.5f---initial time \n* xb0 = %0.5f---initial bore position\n* uN  = %0.5f---front velocity\n* xi  = %0.5f---center of initial current\n* V   = %0.5f---volume of right current\n* xc  = %0.5f---collision point\n* cC  = %0.5f---initial concentration\n'%(t0,xC,uN,xI,VR,xC,cC) + '*'*43 + '\n')

        h_plus  = lambda x_N     : VR/(2*(x_N-xI)) # V here is the initial volume of the right current, NOT the dyanmic V_plus
        u_plus  = lambda x_N,x_B : uN/(x_N-xI)*(x_B-xI)
        h_minus = lambda x_N,x_B : h_plus(x_N)*(1 + (x_N-(2*xI-xC))/(x_B-xC))*hmf
        V_plus  = lambda x_N,x_B : h_plus(x_N)*(x_N-x_B)
        V_minus = lambda x_N,x_B : h_minus(x_N,x_B)*(x_B-xC)

        dSdt  = lambda h_P,h_M,u_P,c_P,c_M         : u_P + np.sqrt((1/2)*(h_M/h_P)*(c_P*h_P*h_P-c_M*h_M*h_M)/(h_P-h_M))
        dcPdt = lambda h_P,c_P                    : -us*c_P/h_P
        dcMdt = lambda h_P,h_M,u_P,c_P,c_M,x_B,x_N : -us*c_M/h_M + (h_P*(c_M-c_P)*(u_plus(x_N,x_B)-dSdt(h_P,h_M,u_P,c_P,c_M)))/V_minus(x_N,x_B)

        def f_rhs(X):
            xb, xn, cm, cp = X

            hP_ = h_plus(xn)
            uP_ = u_plus(xn,xb)
            dcP = dcPdt(hP_,cp)
            if np.abs(xb-xC)>tol:
                hM_ = h_minus(xn,xb)
                ub  = dSdt(hP_,hM_,uP_,cp,cm)
                dcM = dcMdt(hP_,hM_,uP_,cp,cm,xb,xn)
            else:
                #import cmath
                #p = -(H + (uN**2)/3)
                #q = 2*(uN**3)/27 - H*uN/6
                #D = (p**3)/27 + (q**2)/4
                #u1 = -q/2 + complex(D)**0.5
                #return np.real(u1**(1/3)-p/(3*u1**(1/3))-uN/3)
                from scipy.optimize import fsolve
                fv   = lambda v: 2*v**2*(v+uN) - hmf*hP_*cC*(v+hmf*v+hmf*uN)
                ub   = fsolve(fv,0.5)[0]
                hM_  = hP_*(1 + uN/ub)
                zeta = -1#(uN-ub)/(uN+ub)
                dcM  = us*cC/(1-zeta)*(zeta/hP_-1/hM_)
                print('\nh_minus(0) = %0.6f\nub(0)  = %0.6f\ndcM(0) = %0.6f'%(hM_,ub,dcM))
            return np.array([ub,uN,dcM,dcP])

        t,xb,xn,cm,cp = t0,xC,2*xI-xC,cC,cC
        X = np.array([xb,xn,cm,cp])
        T, xB, xN, B_v,cM,cP = [], [], [], [],[],[]
        hM, hP, uP = [], [], []
        while t<=min(final,self.T[-1]-self.coll_time):
            xB.append(X[0])
            cM.append(X[2])
            cP.append(X[3])
            T.append(t)

            k1 = f_rhs(X)
            k2 = f_rhs(X+dt/2*k1)
            k3 = f_rhs(X+dt/2*k2)
            k4 = f_rhs(X+dt*k3)
            U = (k1+2*k2+2*k3+k4)/6
            X += dt*U

            hP_ = h_plus(X[1])
            uP_ = u_plus(X[1],X[0])
            hM_ = h_minus(X[1],X[0])
            hP.append(hP_)
            hM.append(hM_)
            uP.append(uP_)

            xN.append(X[1])
            B_v.append(U[0])

            t+=dt

        return np.array(T),np.array(xN),np.array(xB),np.array(hP),np.array(hM),np.array(uP),np.array(cM),np.array(cP),np.array(B_v)

    def RH_plots(self, show=True,cutoff_init=0,shape_factor=1.0):
        article_params()
        plt.figure(figsize = [7.5,7.5])

        self.bore_data()
        t_num,xN,xB,hP,hM,uP,cM,cP,B_v = self.settling_RH_model(hmf=shape_factor)

        plt.subplot(221)
        plt.plot(self.t_post,self.hP_data,label = 'SW')
        plt.plot(t_num[cutoff_init:],hP[cutoff_init:],label = 'Box')
        plt.xlabel('time')
        plt.ylabel('$h^+$')
        plt.legend()
        plt.subplot(222)
        plt.plot(self.t_post,self.hM_data,label = 'SW')
        plt.plot(t_num[cutoff_init:],hM[cutoff_init:],label = 'Box')
        plt.xlabel('time')
        plt.ylabel('$h^-$')
        plt.legend()
        plt.subplot(223)
        plt.plot(self.t_post,self.uP_data,label = 'SW')
        plt.plot(t_num[cutoff_init:],uP[cutoff_init:],label = 'Box')
        plt.xlabel('time')
        plt.ylabel('$u^+$')
        plt.legend()
        plt.subplot(224)
        plt.plot(self.t_post,self.vel(self.bore),label = 'SW')
        plt.plot(t_num[cutoff_init:],B_v[cutoff_init:],label = 'Box')
        plt.xlabel('time')
        plt.ylabel("$\\frac{dx_b}{dt}$")
        plt.legend()

        plt.subplots_adjust(
            left = 0.08,
            right = 0.99,
            top = 0.99,
            bottom = 0.07,
            hspace = 0.15,
            wspace = 0.2
        )

        if show: plt.show()

    def vel(self,x,sigma=8):
        def interpolate_expand_filter_cut(T,U,sigma,n=25):
            from scipy.ndimage import gaussian_filter1d
            dt = T[1]-T[0]

            tb, ub = T[:n], U[:n]
            te, ue = T[-4*n:], U[-4*n:]
            Pb, Pe = np.polyfit(tb,ub,2), np.polyfit(te,ue,1)

            Tb = np.flip(np.array([T[0]  - (i+1)*dt for i in range(n)]))
            Te = np.array([T[-1] + (i+1)*dt for i in range(n)])
            Ub = Pb[0]*Tb**2 + Pb[1]*Tb + Pb[2]
            Ue = Pe[0]*Te + Pe[1]

            T_expand, U_expand  = np.hstack((Tb,T,Te)), np.hstack((Ub,U,Ue))
            U_filter = gaussian_filter1d(U_expand,sigma = sigma,mode = 'nearest')

            return U_filter[n:-n]
        '''
        This function computes the velocity at the nodes given the poistion.
        The interior points use centered differences, the endpoints use one-sided differences.
        This takes into account the grid not bein equidistant
        '''
        #dt = self.t_post[1]-self.t_post[0]
        dt0 = self.t_post[1:-1]-self.t_post[:-2]
        dt1 = self.t_post[2:]-self.t_post[1:-1]
        v = (dt0/(dt1*(dt0+dt1)))*x[2:] + ((dt1-dt0)/(dt0*dt1))*x[1:-1] + (-dt1/(dt0*(dt0+dt1)))*x[:-2]

        v0 = (-3/2*x[0]+2*x[1]-1/2*x[2])/(self.t_post[1]-self.t_post[0])

        dtN = self.t_post[-1]-self.t_post[-2]
        DtN = self.t_post[-1]-self.t_post[-3]
        a = dtN/(DtN*(DtN-dtN))
        b = -DtN/(dtN*(DtN-dtN))
        c = -a-b
        vN = a*x[-3] + b*x[-2] + c*x[-1]

        print('This velocity data has been filtered with a Gaussian filter with Sigma = %i'%sigma)
        #return gaussian_filter1d(np.hstack((v0,v,vN)),sigma,mode=mode)
        return interpolate_expand_filter_cut(self.t_post,np.hstack((v0,v,vN)),sigma)

    def height_box_data(self,desired_time):
        self.bore_data()
        idx = np.argmin(np.abs(self.t_post - (desired_time-self.coll_time)))
        bore_pos = self.bore[idx]
        front_pos = self.front_data[idx]
        hp = self.hP_data[idx]
        hm = self.hM_data[idx]
        up = self.uP_data[idx]
        um = self.uM_data[idx]
        x_upper_bound = np.ceil(1.05*front_pos)

        article_params()
        plt.figure(figsize=[6.5,1.3])
        plt.subplot(121)
        self.plot_time('h', desired_time, xlim=[0,x_upper_bound])
        plt.plot([bore_pos]*2,[0, hm],color = 'k',linestyle = 'dashed',linewidth = 2)
        plt.plot([0, bore_pos],2*[hm],color = 'k',linestyle = 'dashed',linewidth = 2)
        plt.plot([0, front_pos],2*[0],color = 'k',linestyle = 'dashed',linewidth = 2)
        plt.plot([bore_pos, front_pos],2*[hp],color = 'k',linestyle = 'dashed',linewidth = 2)
        plt.plot([front_pos]*2,[0, hp],color = 'k',linestyle = 'dashed',linewidth = 2)

    def height_box_model(self,desired_time,shape_factor=1.0,mp4=False,SWE_LC = None):
        t_num,xN,xB,hP,hM,uP,cM,cP,B_vel = self.settling_RH_model(hmf=shape_factor)
        idx = np.argmin(np.abs(t_num - (desired_time-self.coll_time)))

        bore_pos  = xB[idx]
        front_pos = xN[idx]
        hp = hP[idx]
        hm = hM[idx]
        x_upper_bound = np.ceil(1.05*front_pos)

        self.plot_time('h', desired_time,color=SWE_LC)
        LC = plt.gca().lines[-1].get_color()
        plt.plot([bore_pos]*2,[0, hm],color = 'k' if mp4 else LC,linestyle = 'dashed',linewidth = 2)
        plt.plot([0, bore_pos],2*[hm],color = 'k' if mp4 else LC,linestyle = 'dashed',linewidth = 2)
        plt.plot([0, front_pos],2*[0],color = 'k' if mp4 else LC,linestyle = 'dashed',linewidth = 2)
        plt.plot([bore_pos, front_pos],2*[hp],color = 'k' if mp4 else LC,linestyle = 'dashed',linewidth = 2)
        plt.plot([front_pos]*2,[0, hp],color = 'k' if mp4 else LC,linestyle = 'dashed',linewidth = 2)

    def box_model_schematic(self,time,show=True,h_pm = 'avg'):
        self.bore_data()
        idx = np.argmin(np.abs(self.t_post - (time-self.coll_time)))
        bore_pos = self.bore[idx]
        front_pos = self.front_data[idx]
        hp = self.hP_data_avg[idx] if h_pm == 'avg' else self.hP_data[idx]
        hm = self.hM_data_avg[idx] if h_pm == 'avg' else self.hM_data[idx]
        up = self.uP_data[idx]
        um = self.uM_data[idx]
        x_upper_bound = min(self.x[-1],np.ceil(1.05*front_pos))

        article_params()
        plt.figure(figsize=[5.125,4])

        # Plot 1, height plot with h+,h-,xN,xB labeled
        plt.subplot(311)
        # self.plot_time('h', time, xlim=[0,x_upper_bound])
        self.plot_time('h', time, xlim=[-x_upper_bound,x_upper_bound])
        AP = dict(facecolor='black',width = 0.1, headwidth = 4,headlength=6) #Arrow properties for annotation
        yb_min = plt.gca().get_ylim()[0] # yb is y bounds, so I want the y bound minimum
        yb_max = plt.gca().get_ylim()[1] # yb is y bounds, so I want the y bound maximum
        plt.annotate(
            '$h_-$',
            xy=(bore_pos+0.02*x_upper_bound,hm),
            xytext=(bore_pos + 0.21*x_upper_bound,hm),
            horizontalalignment='center',
            verticalalignment = 'center',
            arrowprops=AP
        )
        plt.annotate(
            '$h_+$',
            xy=(bore_pos-0.02*x_upper_bound,hp),
            xytext=(bore_pos - 0.21*x_upper_bound,hp),
            horizontalalignment='center',
            verticalalignment = 'center',
            arrowprops=AP
        )
        # plt.annotate(
        #     '$x_b$',
        #     xy=(bore_pos, yb_min),
        #     xytext=(bore_pos, yb_min-0.25*(yb_max-yb_min)),
        #     horizontalalignment='center',
        #     verticalalignment = 'top',
        #     arrowprops=AP
        # )
        # plt.annotate(
        #     '$x_N$',
        #     xy=(front_pos,yb_min),
        #     xytext=(front_pos,yb_min-0.25*(yb_max-yb_min)),
        #     horizontalalignment='center',
        #     verticalalignment = 'top',
        #     arrowprops=AP
        # )
        panel_label(plt.gca())

        # Plot 2, velocity plot with u+,u-,xN,xB labeled
        plt.subplot(312)
        # self.plot_time('u', time, xlim=[0,x_upper_bound])
        self.plot_time('u', time, xlim=[-x_upper_bound,x_upper_bound])
        plt.xlabel('$x$')
        yb_min = plt.gca().get_ylim()[0]
        yb_max = plt.gca().get_ylim()[1] # yb is y bounds, so I want the y bound maximum
        plt.annotate(
            '$u_+$',
            xy=(bore_pos+0.02*x_upper_bound,up),
            xytext=(bore_pos + 0.21*x_upper_bound,up),
            horizontalalignment='center',
            verticalalignment = 'center',
            arrowprops=AP
        )
        plt.annotate(
            '$u_-$',
            xy=(bore_pos-0.02*x_upper_bound,um),
            xytext=(bore_pos - 0.21*x_upper_bound,um),
            horizontalalignment='center',
            verticalalignment = 'center',
            arrowprops=AP
        )
        # plt.annotate(
        #     '$x_b$',
        #     xy=(bore_pos, yb_min),
        #     xytext=(bore_pos, yb_min-0.25*(yb_max-yb_min)),
        #     horizontalalignment='center',
        #     verticalalignment = 'top',
        #     arrowprops=AP
        # )
        # plt.annotate(
        #     '$x_N$',
        #     xy=(front_pos,yb_min),
        #     xytext=(front_pos,yb_min-0.25*(yb_max-yb_min)),
        #     horizontalalignment='center',
        #     verticalalignment = 'top',
        #     arrowprops=AP
        # )
        panel_label(plt.gca())

        # Plot 3, schematic for box model with "ghost" box drawn
        plt.subplot(313)
        self.plot_time('h', time, xlim=[-x_upper_bound,x_upper_bound])
        yb_min = plt.gca().get_ylim()[0]
        yb_max = plt.gca().get_ylim()[1] # yb is y bounds, so I want the y bound maximum
        plt.annotate(
            '$x_b$',
            xy=(bore_pos, yb_min),
            xytext=(bore_pos, yb_min-0.25*(yb_max-yb_min)),
            horizontalalignment='center',
            verticalalignment = 'top',
            arrowprops=AP
        )
        plt.annotate(
            '$x_N$',
            xy=(front_pos,yb_min),
            xytext=(front_pos,yb_min-0.25*(yb_max-yb_min)),
            horizontalalignment='center',
            verticalalignment = 'top',
            arrowprops=AP
        )
        plt.annotate(
            '$2X_i-x_N$',
            xy=(self.apart-front_pos,yb_min),
            xytext=(self.apart-front_pos,yb_min-0.25*(yb_max-yb_min)),
            horizontalalignment='center',
            verticalalignment = 'top',
            arrowprops=AP
        )
        L1,L2 = self.get_oneSided_boxModel_vertices(
            time,
            self.t_post,
            self.front_data,
            self.bore,
            self.hP_data_avg if h_pm == 'avg' else self.hP_data,
            self.hM_data_avg if h_pm == 'avg' else self.hM_data
        )
        plt.plot(L1[0],L1[1],linestyle='dashed',linewidth=2,color='k')
        plt.plot(L2[0],L2[1],linestyle='dashed',linewidth=2,color='k')
        plt.plot([self.coll_loc]*2,[0,hm],linestyle = 'dashed',linewidth=1,color='k')
        plt.plot(
            [self.coll_loc,self.apart-front_pos,self.apart-front_pos,bore_pos],
            [0,0,hp,hp],
            linestyle = 'dashed',
            linewidth=1,
            color='k'
        )
        yRad = 0.75*(hm-hp)
        window_ratio = plt.gca().get_window_extent().height/plt.gca().get_window_extent().width # ratio of the axis window for the y axis to the x axis
        data_ratio = plt.gca().get_data_ratio() # ratio of the data range for the y axis to the x axis. 
        xRad = yRad*window_ratio/data_ratio
        ghost_box_color = 'tab:red'
        plt.gca().add_patch(
            Rectangle(
                (self.apart-front_pos,0),
                front_pos-self.apart,
                hp,
                alpha=0.7,
                color=ghost_box_color
            )
        )
        plt.gca().add_patch(
            Rectangle(
                (self.coll_loc,hp),
                bore_pos-self.coll_loc,
                hm-hp,
                alpha=0.7,
                color=ghost_box_color
            )
        )
        plt.gca().add_patch(
            FancyArrowPatch(
                (self.coll_loc-xRad,hp),
                (self.coll_loc,hp+yRad),
                connectionstyle="arc3,rad=-0.55",
                arrowstyle="simple,head_width=6,head_length=8",
                linewidth=2,
                color=ghost_box_color
            )
        )
        plt.gca().set_xticks([-12,-6,0,6,12])
        panel_label(plt.gca())

        # plt.subplots_adjust(left = 0.10,right = 0.99, bottom = 0.11, top = 0.98,hspace = 0.45)
        plt.subplots_adjust(left = 0.10,right = 0.99, bottom = 0.11, top = 0.98,hspace = 0.0)
        if show:
            plt.show()
        else:
            plt.savefig(self.rootFile + 'solutions/plots/' + 'RhModel_' + self.fileName + '.png',dpi = 1200)
            plt.savefig(self.rootFile + 'solutions/plots/' + 'RhModel_' + self.fileName + '.pdf')
    def num_val_schematic(self,time,show=True):
        self.bore_data()
        idx = np.argmin(np.abs(self.t_post - (time-self.coll_time)))
        bore_pos = self.bore[idx]
        front_pos = self.front_data[idx]
        hp = self.hP_data[idx]
        hm = self.hM_data[idx]
        up = self.uP_data[idx]
        um = self.uM_data[idx]

        x_u = np.ceil(1.05*front_pos)
        x_l = np.floor(1.05*self.x[np.argwhere(self.h[np.argmin(np.abs(self.T-time)),:]>2*self.h_min)[0][0]])
        x_upper_bound =  max(x_u,np.abs(x_l))
        x_lower_bound = -max(x_u,np.abs(x_l))

        article_params()
        plt.figure(figsize=[5.125,2.25])

        #plt.subplot(131)
        ##self.plot_height_conc_time(0,xlim=[-5,5],cb=False)
        #self.plot_time('h', 0, xlim=[-5,5])
        #y_height = plt.gca().get_ylim()[1]-plt.gca().get_ylim()[0]
        #x_width = plt.gca().get_xlim()[1]-plt.gca().get_xlim()[0]
        #plt.annotate(
        #    '$h_r$',
        #    xy=(self.apart/2,self.hR0),
        #    xytext=(self.apart/2 - 0.2*x_width,self.hR0),
        #    horizontalalignment='right',
        #    verticalalignment = 'center',
        #    arrowprops=dict(facecolor='black',width = 1, headwidth = 6,headlength=8)
        #)
        ##plt.annotate('$c_r$',xy=(self.apart/2,0.4*self.hR0),xytext=(self.apart/2 + 0.4*x_upper_bound,0.4*self.hR0), horizontalalignment='left',verticalalignment = 'center',arrowprops=dict(facecolor='black',width = 1, headwidth = 6,headlength=8))
        #plt.text(plt.gca().get_xlim()[0]+0.05*x_width,plt.gca().get_ylim()[0] + 0.8*y_height,'$t$=%i'%0)
        #plt.gca().set_xticks([-5,0,5])

        AP = dict(facecolor='black',width = 0.1, headwidth = 4,headlength=6)
        plt.subplot(2,1,1)
        #self.plot_height_conc_time(time,xlim=[x_lower_bound,x_upper_bound],cb=False)
        self.plot_time('h', time, xlim=[x_lower_bound,x_upper_bound])
        plt.xlabel('')
        plt.ylabel(variable_dict['h' + '_latex'])

        x_width = plt.gca().get_xlim()[1]-plt.gca().get_xlim()[0]
        y_height = plt.gca().get_ylim()[1]-plt.gca().get_ylim()[0]
        plt.annotate(
            '$h_-$',
            xy=(bore_pos+0.01*x_width,hm),
            xytext=(bore_pos + 0.08*x_width,hm),
            horizontalalignment='center',
            verticalalignment = 'center',
            arrowprops=AP
        )
        plt.annotate(
            '$h_+$',
            xy=(bore_pos-0.01*x_width,hp),
            xytext=(bore_pos - 0.08*x_width,hp+0.0),
            horizontalalignment='center',
            verticalalignment = 'center',
            arrowprops=AP
        )
        plt.annotate(
            '$x_b$',
            xy=(bore_pos, hp-0.05*y_height),
            xytext=(bore_pos,plt.gca().get_ylim()[0]-0.25*y_height),
            horizontalalignment='center',
            verticalalignment = 'top',
            arrowprops=AP
        )
        plt.annotate(
            '$x_N$',
            xy=(front_pos,plt.gca().get_ylim()[0]),
            xytext=(front_pos,plt.gca().get_ylim()[0]-0.25*y_height),
            horizontalalignment='center',
            verticalalignment = 'top',
            arrowprops=AP
        )
        panel_label(plt.gca())
        # plt.text(
        #     x_lower_bound+0.05*(x_upper_bound-x_lower_bound),
        #     plt.gca().get_ylim()[0] + 0.8*y_height,
        #     '$t$=%i'%time
        # )
        #plt.gca().set_xticks([-10,0,10])
        plt.gca().set_xticks([-12,-6,0,6,12])
        plt.gca().set_xticklabels(['']*0)

        plt.subplot(2,1,2)
        self.plot_time('u', time, xlim=[x_lower_bound,x_upper_bound])
        plt.ylabel(variable_dict['u' + '_latex'])
        plt.xlabel('$x$')
        plt.annotate(
            '$u_+$',
            xy=(bore_pos+0.01*x_width,up),
            xytext=(bore_pos + 0.08*x_width,up),
            horizontalalignment='center',
            verticalalignment = 'center',
            arrowprops=AP
        )
        plt.annotate(
            '$u_-$',
            xy=(bore_pos-0.01*x_width,um),
            xytext=(bore_pos - 0.08*x_width,um),
            horizontalalignment='center',
            verticalalignment = 'center',
            arrowprops=AP
        )
        vel_bound = np.max(np.abs(plt.gca().get_ylim()))
        plt.ylim([-vel_bound,vel_bound])
        y_height = plt.gca().get_ylim()[1]-plt.gca().get_ylim()[0]
        plt.annotate(
            '$x_b$',
            xy=(bore_pos,up-0.05*y_height),
            xytext=(bore_pos,up-0.35*y_height),
            horizontalalignment='center',
            verticalalignment = 'top',
            arrowprops=AP
        )
        # plt.text(
        #     x_lower_bound+0.05*(x_upper_bound-x_lower_bound),
        #     plt.gca().get_ylim()[0] + 0.8*y_height,
        #     '$t$=%i'%time
        # )
        #plt.gca().set_xticks([-10,0,10])
        plt.gca().set_xticks([-12,-6,0,6,12])
        plt.annotate(
            '$x_N$',
            xy=(front_pos,0-0.05*y_height),
            xytext=(front_pos,plt.gca().get_ylim()[0]-0.15*y_height),
            # xy=(front_pos,up-0.05*y_height),
            # xytext=(front_pos,up-0.35*y_height),
            horizontalalignment='center',
            verticalalignment = 'top',
            arrowprops=AP
        )
        panel_label(plt.gca())

        plt.subplots_adjust(left = 0.11,right = 0.99, bottom = 0.2, top = 0.96,hspace = 0.26)
        plt.savefig(self.rootFile + 'solutions/plots/' + 'NumValSchem_' + self.fileName + '.png',dpi = 1200)
        plt.savefig(self.rootFile + 'solutions/plots/' + 'NumValSchem_' + self.fileName + '.pdf')
        if show: plt.show()

    def spacetime(self,xlim=[None,None],tmax = None,show=False):
        from parmat import cm_data
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('mypar', cm_data, N=256)

        article_params()
        plt.figure(figsize=[2.5,2])
        x_min_idx = np.argmin(np.abs(self.x-xlim[0])) if xlim[0] else 0
        x_max_idx = np.argmin(np.abs(self.x-xlim[1])) if xlim[1] else self.N
        t_idx     = np.argmin(np.abs(self.T-tmax))    if tmax    else len(self.T)
        fig = plt.pcolormesh(self.x[x_min_idx:x_max_idx],self.T[:t_idx],self.h[:t_idx,x_min_idx:x_max_idx],shading = 'gouraud',cmap=cmap,rasterized=True)
        plt.colorbar(fig)
        plt.xlabel('$x$')
        plt.ylabel('$t$')

        plt.subplots_adjust(left = 0.14,right = 0.99, top = 0.97, bottom = 0.19)
        fullFileName = self.rootFile + 'solutions/plots/' + 'SpaceTime_' + self.fileName
        plt.savefig(fullFileName + '.png',dpi = 1200)
        plt.savefig(fullFileName + '_tmp' + '.pdf',dpi = 1200)
        os.replace(fullFileName + '_tmp' + '.pdf',fullFileName + '.pdf')
        if show: plt.show()

    def front_vel(self):
        vel = np.max(self.u[:,np.argwhere(self.x>0)],1)
        self.uN = np.average(vel[self.T>self.coll_time])
        return vel

    def front_vel_plot(self, show=True):
        plt.figure(figsize=[3.5,2])
        article_params()
        vel = self.front_vel()
        plt.plot(self.T,vel)
        plt.axvspan(self.coll_time,self.T[-1],color = 'gray', alpha = 0.3)
        plt.plot(self.T[self.T>self.coll_time], self.T[self.T>self.coll_time]*0+self.uN, linestyle = 'dashed',color='k')
        plt.text(3*self.coll_time, 0.98*self.uN, '$u_N\\approx%0.2f$'%self.uN, verticalalignment='top', horizontalalignment='left')

        plt.xlabel('time, $t$')
        plt.ylabel('nose velocity, $\\frac{dx_N}{dt}$')
        plt.text((self.T[-1]+self.T[0])/2, (vel[-1]+vel[0])/2, 'post collision', verticalalignment='center', horizontalalignment='center')

        plt.xlim([None,self.T[-1]])
        plt.subplots_adjust(left = 0.16,bottom = 0.22, top = 0.98, right = 0.98)
        plt.savefig(self.rootFile + 'solutions/plots/' + 'FrontVel_' + self.fileName + '.png',dpi = 1200)
        plt.savefig(self.rootFile + 'solutions/plots/' + 'FrontVel_' + self.fileName + '.pdf')
        if show: plt.show()

    def equal_conc(self,desired_time):
        T_idx = np.argmin(np.abs(self.T-desired_time))
        C = self.c1[T_idx,:] - self.c2[T_idx,:]
        low_idx,high_idx = np.argmax(C),np.argmin(C)
        C_local = C[low_idx:high_idx]
        x_local = self.x[low_idx:high_idx]

        return intersection(x_local,C_local,x_local*0)
    def equal_conc_pos_vs_time(self,start_time,label=None):
        times = self.T[self.T>=start_time]
        EC = np.array([self.equal_conc(t) for t in times])
        plt.plot(times-self.coll_time,EC-self.coll_loc,label = label)

        plt.xlabel('time (t=0 is collision)')
        plt.ylabel('position of equal concentration (0 is collision)')
    def compare_conc(self,shape_factor=1.):
        article_params()
        self.bore_data()
        t_num,xN,xB,hP,hM,uP,cM,cP,B_vel  = self.settling_RH_model(hmf = shape_factor)
        plt.plot(self.t_post,self.cM_data_avg,label='average $c_-$, SW',linewidth =1)
        plt.plot(self.t_post,self.cP_data_avg,label='average $c_+$, SW',linewidth =1)
        plt.plot(t_num,cM,linestyle='dashed',color='tab:blue',  label='$c_-$, Box',linewidth =1)
        plt.plot(t_num,cP,linestyle='dashed',color='tab:orange',label='$c_+$, Box',linewidth =1)
        plt.legend()
        plt.title('settling, $u_s=$%0.2f'%self.U_s)
        plt.xlabel('time')
        plt.ylabel('concentration')

def Deposit_Results(SimVars = [(0.70, 0.70),(0.70, 1.00),(1.00, 0.70),(1.00, 1.00),(0.70, 1.43),(1.43, 0.70)],SimPack=None,U_s=0.01,rootFile = 'Mar3_DepositionExamplePlots/',N=28000,sharp=200,finalTime=40.,save=True):

    if SimPack == None:
        Sims = []
        NoCollSims = []
        LeftCurr = TurbiditySim(1.0,1.0,U_s,rootFile,['d2'], subFile='sims/OneCurrOnly_', sharp=sharp, N=N, finalTime=finalTime)
        for sim in SimVars:
            Sims.append(TurbiditySim(sim[0],sim[1],U_s,rootFile,['d1','d2'], sharp=sharp, N=N, finalTime=finalTime))
            NoCollSims.append(TurbiditySim(sim[0],sim[1],U_s,rootFile,['d2'], subFile='sims/OneCurrOnly_', sharp=sharp, N=N, finalTime=finalTime))
    else:
        Sims,NoCollSims,LeftCurr = SimPack
    LeftCurr_d1 = np.flip(LeftCurr.d2[-1,:])

    yMax = 0
    for sim in Sims:
        D = sim.d1+sim.d2
        yMax = max(yMax,np.max(D))
    print(yMax)

    for coll,no_coll in zip(Sims,NoCollSims):
        coll.plot_deposit_gradient(dt_plot=0.1,show=False,save=False,xb=20,yb=yMax,close=False,title=True)
        plt.plot(no_coll.x, no_coll.d2[-1,:] + LeftCurr_d1, label = 'No Collision')
        plt.legend()
        plt.subplots_adjust(left = 0.1, right =0.99,bottom=0.23,top =0.84)
        if save: plt.savefig(coll.rootFile + 'solutions/plots/gradientDeposit_' + coll.fileName + '.png',dpi=1000)
        #if save: plt.savefig(coll.rootFile + 'solutions/plots/gradientDeposit_' + coll.fileName + '.pdf',dpi=1000)
        plt.close()
    return Sims,NoCollSims,LeftCurr

def Box_SWE_Asym(SimVars=[(1.0,1.0),(1.06,0.85),(1.11,0.7)],Sims=None,sharp=100,N=7000,finalTime=40.,shape_factor=1.):
    '''
    This function plots the position and the velocity of the bore
    for both the Box Model and Shallow Water Model.
    Here, settling is assumed to be 0, and the asymmetry is varied.
    in Box_SWE_Settling(), the settling is nonzero, but the currents are symmetric.
    '''
    article_params()
    plt.rcParams.update({'lines.linewidth':1})
    if Sims == None:
        Sims = []
        for sim in SimVars:
            Sims.append(TurbiditySim(sim[0],sim[1],0.0,'Nov24_AsymBoxModel/',['h','u','c1','c2'], sharp=sharp, N=N, finalTime=finalTime))

    plt.figure(figsize=[5.125,3])
    legend_list = ['SWE','Box Model','']
    legend_colors = ['k']*2 + ['white']
    legend_linestyles = list(mpllinestyles[:2]) + ['solid']
    for i,sim in enumerate(Sims):
        sim.bore_data()
        t_num,xN,xB,hP,hM,uP,cM,cP,B_vel = sim.settling_RH_model(hmf=shape_factor)
        plt.subplot(211)
        plt.plot(sim.t_post,sim.vel(sim.bore),color=tabcolors[i],linestyle = mpllinestyles[0])
        plt.plot(t_num,B_vel,color=tabcolors[i],linestyle = mpllinestyles[1])
        # plt.xlabel('time, $t$')
        plt.ylabel("velocity, $\\frac{dx_b}{dt}$")
        plt.xlim([0,None])
        plt.ylim([0,None])

        plt.subplot(212)
        plt.plot(sim.t_post,sim.bore,color=tabcolors[i],linestyle = mpllinestyles[0])
        plt.plot(t_num,xB,color=tabcolors[i],linestyle = mpllinestyles[1])
        plt.xlabel('time post-collision, $t-t_c$')
        plt.ylabel('position, $x_b$')
        plt.xlim([0,None])
        plt.ylim([0,None])
        legend_list.append('$h_0$=%0.2f, $c_0$=%0.2f'%(sim.hR0,sim.cR0))

    legend_colors += list(tabcolors[:len(Sims)])
    legend_linestyles += [mpllinestyles[0]]*(len(Sims))
    #for sp in [2,1]:
    #    plt.subplot(2,1,sp)
    leg = plt.legend(legend_list,ncol=1,loc='upper center', bbox_to_anchor=(1.25,1.6))
    for i,j in enumerate(leg.legend_handles):
        j.set_color(legend_colors[i])
        j.set_linestyle(legend_linestyles[i])
    plt.subplots_adjust(left = 0.11, top = 0.98, bottom = 0.14, right = 0.7, hspace = 0.18)
    plt.savefig(sim.rootFile + 'solutions/plots/' + f'BoxSW_DiffHC_shapefactor{shape_factor:0.2}'.replace('.','_') + '.pdf')
    plt.savefig(sim.rootFile + 'solutions/plots/' + f'BoxSW_DiffHC_shapefactor{shape_factor:0.2}'.replace('.','_') + '.png',dpi=600)
    plt.close()
    return Sims

def Box_SWE_Settling(U_s=[0,0.01,0.02],Sims=None,sharp=100,N=14000,finalTime=40.,shape_factor=1.,dt=0.01):
    '''
    This function plots the position and the velocity of the bore
    for both the Box Model and Shallow Water Model.
    Here, settling is nonzero, but the currents are symmetric,
    in Box_SWE_Asym(), the settling is 0, but the currents are asymmetric.
    '''
    article_params()
    plt.rcParams.update({'lines.linewidth':1})
    if Sims == None:
        Sims = []
        for us in U_s:
            Sims.append(TurbiditySim(1.0,1.0,us,'Nov24_AsymBoxModel/',['h','u','c1','c2'], sharp=sharp, N=N, finalTime=finalTime))

    plt.figure(figsize=[5.125,3.])
    legend_list = ['SWE','Box Model','']
    legend_colors = ['k']*2 + ['white']
    legend_linestyles = list(mpllinestyles[:2]) + ['solid']
    for i,sim in enumerate(Sims):
        sim.bore_data()
        t_num,xN,xB,hP,hM,uP,cM,cP,B_vel = sim.settling_RH_model(hmf=shape_factor,dt=dt)
        plt.subplot(211)
        plt.plot(sim.t_post,sim.vel(sim.bore),color=tabcolors[i],linestyle = mpllinestyles[0])
        plt.plot(t_num,B_vel,color=tabcolors[i],linestyle = mpllinestyles[1])
        # plt.xlabel('time post-collision, $t-t_c$')
        plt.ylabel("velocity, $\\frac{dx_b}{dt}$")
        plt.ylim([0,None])
        plt.xlim([0,None])

        plt.subplot(212)
        plt.plot(sim.t_post,sim.bore,color=tabcolors[i],linestyle = mpllinestyles[0])
        plt.plot(t_num,xB,color=tabcolors[i],linestyle = mpllinestyles[1])
        plt.xlabel('time post-collision, $t-t_c$')
        plt.ylabel('position, $x_b$')
        plt.xlim([0,None])
        plt.ylim([0,None])
        legend_list.append('$U_s=$%0.2f'%(sim.U_s))

    legend_colors += list(tabcolors[:len(Sims)])
    legend_linestyles += [mpllinestyles[0]]*(len(Sims))
    leg = plt.legend(legend_list,ncol=1,loc='upper center', bbox_to_anchor=(1.195,1.6))
    for i,j in enumerate(leg.legend_handles):
        j.set_color(legend_colors[i])
        j.set_linestyle(legend_linestyles[i])
    plt.subplots_adjust(left = 0.11, top = 0.98, bottom = 0.14, right = 0.7, hspace = 0.18)
    plt.savefig(sim.rootFile + 'solutions/plots/' + f'BoxSW_Settling_DiffHC_shapefactor{shape_factor:0.2}'.replace('.','_') + '.pdf')
    plt.savefig(sim.rootFile + 'solutions/plots/' + f'BoxSW_Settling_DiffHC_shapefactor{shape_factor:0.2}'.replace('.','_') + '.png',dpi=600)
    plt.close()

    plt.figure(figsize=[5.125,2.])
    legend_list = ['$c_-$','$c_+$','']
    legend_colors = ['k']*2 + ['white']
    legend_linestyles = list(mpllinestyles[:2]) + ['solid']
    for i,sim in enumerate(Sims):
        sim.bore_data()
        t_num,xN,xB,hP,hM,uP,cM,cP,B_vel = sim.settling_RH_model(hmf=shape_factor,dt=dt)
        plt.plot(t_num,cM,color=tabcolors[i],linestyle = mpllinestyles[0])
        plt.plot(t_num,cP,color=tabcolors[i],linestyle = mpllinestyles[1])
        plt.xlabel('time, $t$')
        plt.ylabel('concentration')

        legend_list.append('$U_s=$%0.2f'%(sim.U_s))

    legend_colors += list(tabcolors[:len(Sims)])
    legend_linestyles += [mpllinestyles[0]]*(len(Sims))
    leg = plt.legend(legend_list,ncol=1,loc='upper center', bbox_to_anchor=(1.15,1.))
    for i,j in enumerate(leg.legend_handles):
        j.set_color(legend_colors[i])
        j.set_linestyle(legend_linestyles[i])
    plt.subplots_adjust(left=0.09,bottom=0.19,top=0.97,right=0.78)
    plt.savefig(sim.rootFile + 'solutions/plots/' + f'BoxSW_Settling_Conc_shapefactor{shape_factor:0.2}'.replace('.','_') + '.pdf')
    plt.savefig(sim.rootFile + 'solutions/plots/' + f'BoxSW_Settling_Conc_shapefactor{shape_factor:0.2}'.replace('.','_') + '.png',dpi=600)
    plt.close()

    plt.figure(figsize=[5.125,2.])
    plt.subplot(121)
    Sims[1].compare_conc()
    plt.subplot(122)
    Sims[2].compare_conc()
    plt.tight_layout()
    plt.savefig(sim.rootFile + 'solutions/plots/' + f'BoxSW_CompareConc_shapefactor{shape_factor:0.2}'.replace('.','_') + '.pdf')
    plt.savefig(sim.rootFile + 'solutions/plots/' + f'BoxSW_CompareConc_shapefactor{shape_factor:0.2}'.replace('.','_') + '.png',dpi=600)
    plt.close()
    return Sims

def suspended_concentration(US, RootFile):
    def print_latex_line(label,l):
        string = label
        for i in l:
            string += ' & %0.3f'%i
        string += ' \\\\'
        print(string)

    no_coll_time, coll_time = [], []
    for i,us in enumerate(US):
         t1 = TurbiditySim(0.0,0.0,us,RootFile,['c1','c2'],N=20000,sharp = 200)
         no_coll_time.append(t1.sus_conc(LS = 'dashed', LC = tabcolors[i]))

         t2 = TurbiditySim(1.0,1.0,us,RootFile,['c1','c2'],N=20000,sharp = 200)
         coll_time.append(t2.sus_conc(LS = 'solid',LC = tabcolors[i]))
    plt.plot(t1.T,t1.T*0+0.05,color = 'k', linewidth = 1)
    plt.xlabel('time')
    plt.ylabel('Percent of suspended concentration')

    print_latex_line('$U_s$',US)
    print_latex_line('no collision',no_coll_time)
    print_latex_line('collision',coll_time)
    legendlist = ['no collision','collision','',''] + ['$U_s = %0.3f$'%us for us in US]
    colors = ['k']*2 + ['white']*2 + list(tabcolors[:len(US)]) 
    linestyles = ['dashed'] + ['solid']*7
    leg = plt.legend(legendlist,ncol=2)#,loc='upper center', bbox_to_anchor=(0.5, -0.23))
    for i,j in enumerate(leg.legend_handles):
        j.set_color(colors[i])
        j.set_linestyle(linestyles[i])
    plt.savefig('suspended_contration.pdf')


def deposit_example_plots(US = [0.005,0.01,0.015]):
    #plt.rcParams.update({"text.usetex":True})
    def full_deposit():
        article_params()
        plt.figure(figsize=[8,2.5])
        subplot_counter = 0
        for h0,c0 in zip([1.0,0.7,1.0],[1.0,1.0,0.7]):
            subplot_counter+=1
            plt.subplot(1,3,subplot_counter)
            for i,us in enumerate(US):
                tt = TurbiditySim(h0,c0,us,testFileName,['d1','d2'])
                tt.plot_deposit(LS = mpllinestyles[i])

        plt.subplot(1,3,2)
        legendlist = ['$U_s = %0.3f$'%us for us in US] + ['$d_1(x)$','$d_2(x)$']
        colors = ['k' for i in range(len(US))] + list(tabcolors[:2])
        leg = plt.legend(legendlist,ncol=len(legendlist),loc='upper center', bbox_to_anchor=(0.5, -0.23))
        linestyles = list(mpllinestyles[:3]) + ['solid' for i in range(2)]
        for i,j in enumerate(leg.legend_handles):
            j.set_color(colors[i])
            j.set_linestyle(linestyles[i])

        plt.subplot(1,3,1)
        plt.ylabel('deposit')
        plt.subplots_adjust(left = 0.07,right = 0.99, bottom = 0.2, top = 0.85,wspace = 0.18)
        plt.savefig(tt.rootFile + 'solutions/plots/deposit_examples' + tt.fileName + '.png', bbox_inches='tight',dpi=400)
    def encroachment_only(us):
        article_params()
        plt.figure(figsize=[8,3])
        subplot_counter = 0
        wyl = True # want y legend? 
        for h0,c0 in zip([1.0,0.7],[1.0,1.0]):
            subplot_counter+=1
            plt.subplot(1,2,subplot_counter)
            tt = TurbiditySim(h0,c0,us,testFileName,['d1','d2'])
            tt.plot_encroachment(wantLegend=False, want_y_label=wyl)
            wyl = False # want y legend? 

        #plt.subplot(1,2,2)
        #legendlist = ['$U_s = %0.3f$'%us for us in US] + ['$d_1(x)$','$d_2(x)$'] 
        #colors = ['k' for i in range(len(US))] + list(tabcolors[:2])
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        #linestyles = list(mpllinestyles[:3]) + ['solid' for i in range(2)]
        #for i,j in enumerate(leg.legend_handles):
        #    j.set_color(colors[i])
        #    j.set_linestyle(linestyles[i])

        plt.subplots_adjust(left = 0.07,right = 0.75, bottom = 0.2, top = 0.85,wspace = 0.18)
        plt.savefig(tt.rootFile + 'solutions/plots/encroachmentOnly_versus_IC_' + tt.fileName + '.png', bbox_inches='tight',dpi=400)
    full_deposit()
    encroachment_only(0.005)
    encroachment_only(0.01)
    encroachment_only(0.015)

def encroachment_by_concentration(t=[(1.0,1.0),(1.04,0.9),(1.06,0.8),(1.11,0.7),(0.7,0.7)], times=[i*10 for i in range(1,5)], U_s=0.0, rootFile='Nov24_AsymBoxModel/',sharp=100,N=7000):
    article_params()
    t = [TurbiditySim(i[0],i[1],U_s,rootFile,['h','u','c1','c2'],sharp=sharp,N=N) for i in t]
    for jj,i in enumerate(t):
        plt.figure()
        plt.title('$h_0$ = %0.2f, $c_0$=%0.2f'%(i.hR0,i.cR0))
        i.plot_times('c1',times,xlim=[-2,2] if jj<len(t)-1 else [0,12])
        plt.gca().set_prop_cycle(None)
        i.plot_times('c2',times,xlim=[-2,2] if jj<len(t)-1 else [0,12],linestyle='dashed',wl=False)
        plt.plot(2*[i.coll_loc],[0,1],color='k')
        plt.ylabel('concentrations \n solid: c1, dashed: c2')
        plt.tight_layout()
        plt.savefig('Nov24_AsymBoxModel/solutions/plots/' + f'ConcentrationAtCollision_h0_{i.hR0:.2f}_c0_{i.cR0:.2f}'.replace('.','_')+'.pdf')
        plt.close()

    plt.figure(figsize=[8,4])
    plt.subplot(121)
    for i in t:
        i.equal_conc_pos_vs_time(2*i.coll_time,label = '$h_0$: %0.2f, $c0$: %0.2f'%(i.hR0,i.cR0))
    plt.legend()
    plt.subplot(122)
    for i in t[:-1]:
        i.equal_conc_pos_vs_time(2*i.coll_time,label = '$h_0$: %0.2f, $c_0$: %0.2f'%(i.hR0,i.cR0))
    plt.legend()
    plt.tight_layout()
    plt.savefig('Nov24_AsymBoxModel/solutions/plots/EqualConcenctrationPosition.pdf')
    plt.close()

    return t

class DepositionAnalysis:
    # SedimentationInitialConditionTest_2025Jun7 is rootFile for full test.
    def __init__(self, U_s, rootFile, H2=np.linspace(0.7,1.42,73), C2=np.linspace(0.7,1.42,73), N = 5000, NuRe = 1000, finalTime = 40., h_min = 0.0001, CFL = 0.1, sharp = 50, apart = 5., FrSquared = 1., hL0 = 1.0, cL0 = 1.0, NuPe = None, subFile = 'solutions/postData/'):
        self.H2 = H2
        self.C2 = C2
        self.H_mesh,self.C_mesh = np.meshgrid(self.H2,self.C2)
        self.U_s = U_s
        self.rootFile = rootFile
        self.subFile = subFile
        self.N = N
        self.NuRe = NuRe
        self.finalTime = finalTime
        self.h_min = h_min
        self.CFL = CFL
        self.sharp = sharp
        self.apart = apart
        self.FrSquared = FrSquared
        self.hL0, self.cL0 = hL0, cL0

        self.fileName = "%iby%i_%iapart_N%i_CFL%0.3f_T%0.1f_NuRe%i_FrFr%0.3f_Us%0.3f_hmin%0.5f_sharp%i"%(self.H2.shape[0],self.C2.shape[0],self.apart,self.N,self.CFL,self.finalTime,self.NuRe,self.FrSquared,self.U_s,self.h_min,self.sharp)

        try:
            self.encroachment_mass = np.loadtxt(self.rootFile + self.subFile + 'encroachMass_' + self.fileName + '.csv', delimiter = ',')
            self.COM_x = np.loadtxt(self.rootFile + self.subFile + 'COMx_' + self.fileName + '.csv', delimiter = ',')
        except OSError:
            print('Cannot find processed sedimentation data. Post processing now.')
            start = time.time()
            self.encroachment_mass = np.zeros([C2.shape[0],H2.shape[0]])
            self.COM_x = np.zeros([C2.shape[0],H2.shape[0]])
            self.deposited_mass = np.zeros([C2.shape[0],H2.shape[0]])
            self.coll_time = np.zeros([C2.shape[0],H2.shape[0]])
            self.coll_loc = np.zeros([C2.shape[0],H2.shape[0]])
            for i,h2 in enumerate(H2):
                for j,c2 in enumerate(C2):
                    if h2*c2<=1:
                        temp_sim = TurbiditySim(h2,c2,self.U_s,self.rootFile,['d1','d2'],N=self.N, NuRe=self.NuRe, finalTime=self.finalTime, h_min=self.h_min, CFL=self.CFL, sharp=self.sharp, apart=self.apart, FrSquared=self.FrSquared)
                        if temp_sim.d1.shape[0]>1 and temp_sim.d2.shape[0]>1:
                            temp_sim.deposition_details()
                            self.encroachment_mass[j,i] = temp_sim.encroachment_mass
                            self.COM_x[j,i] = temp_sim.COM_x
                            self.deposited_mass[j,i] = (np.sum(temp_sim.d1)+np.sum(temp_sim.d2))*(temp_sim.x[1]-temp_sim.x[0])/(temp_sim.hL0*temp_sim.cL0 + h2*c2)
                            self.coll_time[j,i] = temp_sim.coll_time
                            self.coll_loc[j,i] = temp_sim.coll_loc
                        else:
                            print('h2 = %0.2f, c2 = %0.2f, U_s = %0.3f simulation did not finish. Final deposit data DNE'%(h2,c2,self.U_s))
                            self.encroachment_mass[j,i] = np.nan
                            self.coll_time[j,i] = np.nan
                            self.coll_loc[j,i] = np.nan
                            self.COM_x[j,i] = np.nan
                    else:
                        self.encroachment_mass[j,i] = np.nan
                        self.COM_x[j,i] = np.nan
                        self.coll_time[j,i] = np.nan
                        self.coll_loc[j,i] = np.nan
            np.savetxt(self.rootFile + self.subFile + 'encroachMass_' + self.fileName + '.csv', self.encroachment_mass, delimiter = ',')
            np.savetxt(self.rootFile + self.subFile + 'COMx_' + self.fileName + '.csv', self.COM_x, delimiter = ',')
            print('%0.2f seconds'%(time.time()-start))

    def myPcolor(self,attr,plotTitle,streamlines=True,save=False):
        if save:
            article_params()
            plt.figure(figsize=[2.6,2.1])
        Z = getattr(self,attr)
        Zm = np.ma.masked_invalid(Z)
        pplot = plt.pcolormesh(self.H2,self.C2,Z,shading = 'gouraud')
        plt.contour(self.H2,self.C2,Z,colors='k',levels=[0])
        plt.gca().set_aspect('equal')
        if plotTitle: plt.title(plotTitle)
        plt.xlabel('$h_r$')
        plt.ylabel('$c_r$')

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(pplot,cax=cax,format=tkr.FormatStrFormatter('%.1f'))
        cb.set_ticks(np.arange(np.ceil(Zm.min()*10),np.floor(Zm.max()*10)+1)/10)
        plt.subplots_adjust(left=0.1,right=0.9,top=0.99,bottom=0.16)
        if save:
            plt.savefig(self.rootFile + 'solutions/plots/' + attr + self.fileName + '.png', bbox_inches='tight',dpi=1000)
            plt.savefig(self.rootFile + 'solutions/plots/' + attr + self.fileName + '.pdf', bbox_inches='tight',dpi=800)
            plt.close()
            plt.rcParams.update({"text.usetex":False})

    def plot_dimensional_analysis(self):
        article_params()
        plt.figure(figsize=[7,2.75])
        plt.subplot(121)
        self.myPcolor('encroachment_mass','Encroachment Mass')
        plt.subplot(122)
        self.myPcolor('COM_x','COM $x$-coordinate')
        plt.tight_layout()

        plt.savefig(self.rootFile + 'solutions/plots/deposition_analysis_' + self.fileName + '.png', bbox_inches='tight',dpi=400)
        plt.close()
        plt.rcParams.update({"text.usetex":False})

    def linear_appr(self,X,Y,Z):
        Xf,Yf,Zf = X.flatten(), Y.flatten(), Z.flatten()
        no_nans = ~np.isnan(Zf)
        Xf,Yf,Zf = Xf[no_nans],Yf[no_nans],Zf[no_nans]
        A = np.array([Xf,Yf,np.ones(Zf.shape[0])]).T
        a, b, c =  np.linalg.solve(np.matmul(A.T,A),np.matmul(A.T,np.array([Zf]).T)).flatten()
        print('a = %0.4f, b = %0.4f, c = %0.4f'%(a,b,c))
        return a*X + b*Y + c*np.ones(Z.shape) + 0*Z #The +0*Z at the end is to ``re-introduce'' the nans so that the approximation is not plotted everywhere (bit overwhelming)

    def quadratic_appr(self,X,Y,Z):
        Xf,Yf,Zf = X.flatten(), Y.flatten(), Z.flatten()
        no_nans = ~np.isnan(Zf)
        Xf,Yf,Zf = Xf[no_nans],Yf[no_nans],Zf[no_nans]
        A = np.array([Xf*Xf,Xf*Yf,Yf*Yf,Xf,Yf,np.ones(Zf.shape[0])]).T
        a, b, c, d, e, f =  np.linalg.solve(np.matmul(A.T,A),np.matmul(A.T,np.array([Zf]).T)).flatten()
        print('a = %0.4f, b = %0.4f, c = %0.4f, d = %0.4f, e = %0.4f, f = %0.4f'%(a,b,c,d,e,f))
        return a*X*X + b*X*Y + c*Y*Y + d*X + e*Y + f*np.ones(Z.shape) + 0*Z #The +0*Z at the end is to ``re-introduce'' the nans so that the approximation is not plotted everywhere (bit overwhelming)

    def mpl3Dplot(self,azim=-50,elev=10,save=True):
        article_params()
        X = self.H_mesh
        Y = self.C_mesh
        Z = self.encroachment_mass
        Z_masked = np.ma.masked_invalid(Z)
        P = self.linear_appr(X,Y,Z)
        P_masked = np.ma.masked_invalid(P)
        fig,ax = plt.subplots(figsize = (2.6,2.1),subplot_kw = dict(projection = '3d'))
        ax.plot_surface(X,Y,Z_masked,cmap=cm['viridis'],vmin=Z_masked.min(),vmax=Z_masked.max(),linewidth=0,alpha=0.7)
        #ax.plot_surface(X,Y,P_masked,cmap=cm['plasma'],vmin=P_masked.min(),vmax=P_masked.max())
        ax.plot_wireframe(X,Y,P_masked,linewidth=0.5,color='k')
        ax.set_xlabel('$h_r$',labelpad=-5)
        ax.set_ylabel('$c_r$',labelpad=-7)
        ax.set_zlabel('Encroachment Mass',labelpad=-3)
        ax.xaxis.set_tick_params(pad=-5)
        ax.yaxis.set_tick_params(pad=-5)
        ax.zaxis.set_tick_params(pad=0)
        ax.azim = azim
        ax.elev = elev
        plt.subplots_adjust(left = -0.15,right=0.95,bottom=0.0,top=1.1)
        if save:
            plt.savefig(self.rootFile + 'solutions/plots/3Dview_' + self.fileName + '.png', bbox_inches='tight', pad_inches=0.25, dpi=1000)
            plt.savefig(self.rootFile + 'solutions/plots/3Dview_' + self.fileName + '.pdf', bbox_inches='tight', pad_inches=0.2)
            plt.close()
        else:
            plt.show()

    def plotly3Dplot(self, azim=-80, elev=22):
        X = self.H_mesh
        Y = self.C_mesh
        Z = self.encroachment_mass
        P = self.linear_appr(X, Y, Z)

        # Mask NaNs
        Zm = np.ma.masked_invalid(Z)
        Pm = np.ma.masked_invalid(P)

        # Convert azim/elev → Plotly camera
        r = 1.4
        az,el = np.deg2rad(azim),np.deg2rad(elev)
        camera = dict(eye=dict(x=r*np.cos(el)*np.cos(az), y=r*np.cos(el)*np.sin(az), z=r*np.sin(el)))

        fig = go.Figure()

        # Surface 1: linear approximation
        fig.add_trace(go.Surface(x=X,y=Y,z=Pm,colorscale="Plasma",showscale=False,opacity=1.0))

        # Surface 2: encroachment mass
        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=Zm,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title="Encroachment Mass",
                    tickfont=dict(size=8)
                )
            )
        )

        fig.update_layout(
            width=1040,   # ~2.6 in at 400 dpi
            height=840,   # ~2.1 in at 400 dpi
            margin=dict(l=10, r=10, t=10, b=10),
            font=dict(
                family="Times New Roman",
                size=10,
                color="black"
            ),
            scene=dict(
                camera=camera,
                xaxis=dict(
                    title=r"$h_r$",
                    showbackground=False,
                    showgrid=True,
                    gridcolor="lightgray",
                    zeroline=False,
                    tickfont=dict(size=8)
                ),
                yaxis=dict(
                    title=r"$c_r$",
                    showbackground=False,
                    showgrid=True,
                    gridcolor="lightgray",
                    zeroline=False,
                    tickfont=dict(size=8)
                ),
                zaxis=dict(
                    title="Encroachment Mass",
                    showbackground=False,
                    showgrid=True,
                    gridcolor="lightgray",
                    zeroline=False,
                    tickfont=dict(size=8)
                ),
                aspectmode="auto"
            ),
            paper_bgcolor="white"
        )

        #fig.show()
        #plt.savefig(self.rootFile + 'solutions/plots/' + attr + self.fileName + '.pdf', bbox_inches='tight',dpi=800)
        fig.write_image(self.rootFile + 'solutions/plots/3Dview_' + self.fileName +'.png')

    def my3Dplot(self):

        encroachment_linear = self.linear_appr(self.H_mesh,self.C_mesh,self.encroachment_mass)
        encroachment_quadratic = self.quadratic_appr(self.H_mesh,self.C_mesh,self.encroachment_mass)

        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{'type':'surface'}]*2],
            subplot_titles=(r'$\text{Encroachment Mass}$', r'$\text{COM }x\text{-coordinate}$'),
            horizontal_spacing=0.15
        )

        fig.add_trace(go.Surface(z=self.encroachment_mass, x=self.H_mesh, y=self.C_mesh, colorscale='viridis',colorbar={'x':0.43,'title':'data'}), row=1, col=1)
        fig.add_trace(go.Surface(z=encroachment_linear, x=self.H_mesh, y=self.C_mesh, colorscale='Plotly3', colorbar={'x':0.51, 'title':'Linear approximation'}), row=1, col=1)

        fig.add_trace(go.Surface(z=self.COM_x, x=self.H_mesh, y=self.C_mesh, colorscale='plasma'), row=1, col=2)

        fig.update_layout(title_text=r'$\text{Settling speed}: U_s=%0.3f$'%self.U_s,
            height = 600, width = 1200,
            scene ={'xaxis_title':r"h_{2,0}",'yaxis_title':r"c_{2,0}",
                'aspectmode':'manual','aspectratio':dict(x=1, y=1, z=1.),'camera':dict(eye=dict(x=1.1, y=1.98, z=0.66))},
            scene2={'xaxis_title':r'h_{2,0}','yaxis_title':r'c_{2,0}',
                'aspectmode':'manual','aspectratio':dict(x=1, y=1, z=1.),'camera':dict(eye=dict(x=1.1, y=1.98, z=0.66))})
        fig.write_html(self.rootFile + 'solutions/plots/3Dview_' + self.fileName +'.html', include_mathjax="cdn")
        del fig

    def get_no_encroachment_data(self):
        '''
        Find the (h0,c0) values where encroachment_mass = 0.
        That is, find the 0 level-set of encroachment_mass.
        This is meant to identify an (h0,c0) pair where no encroachment occurs, so I stored this as a dictionary, like a look up table.
        '''
        print('\n!!!  This will return a dictionary with the key being c0 and the value being h0, i.e., no_encroachment[c0]=h0  !!! \n')
        C2_below_1 = np.round(self.C2[:np.argwhere(self.C2<=1)[-1][0]+1],2)
        self.no_encroachment = {}
        for i in range(len(C2_below_1)):
            p_idx = np.argwhere(self.encroachment_mass[i,:]>0)[-1][0] # p_idx is for positive index, the last positive value. This is dependent on the encroachment mass being a descending function of initial height for a fixed initial c. 
            self.no_encroachment[C2_below_1[i]] = np.round((self.H2[p_idx]+self.H2[p_idx+1])/2,2)

def make_deposition_plots(US=[0.005,0.01,0.015]):
    '''
    This function runs creates plot when you want to plot multiple surface for differnt settling speeds at once.
    The DepositionAnalysis class has to be called for each settling speed to generate the data.
    '''
    def singleSettlingSpeeds():
        for u in US:
            d = DepositionAnalysis(u,'SedimentationInitialConditionTest_2025Jun7/')
            d.my3Dplot()
            d.plot_dimensional_analysis()

    def layeredSettlingSpeeds():
        fig = make_subplots(rows=1, cols=2,
            specs=[[{'type':'surface'}]*2],
            subplot_titles=(r'$\text{Encroachment Mass}$', r'$\text{COM }x\text{-coordinate}$'))

        settlingSpeedCM = ['Blugrn','Plotly3','Burg']

        for i,u in enumerate(US):
            d = DepositionAnalysis(u,'SedimentationInitialConditionTest_2025Jun7/')


            fig.add_trace(go.Surface(z=d.encroachment_mass, x=d.H_mesh, y=d.C_mesh,
                colorscale=settlingSpeedCM[i],
                colorbar={'x':0.29+0.07*i,'title':{'text':'$U_s=%0.3f$'%(u),'side':'top'},'len':0.75,'thickness':20}),
                row=1, col=1)
            fig.add_trace(go.Surface(z=d.COM_x, x=d.H_mesh, y=d.C_mesh,
                colorscale=settlingSpeedCM[i],
                colorbar={'x':0.79+0.07*i,'title':{'text':'$U_s=%0.3f$'%(u),'side':'top'},'len':0.75,'thickness':20}), 
                row=1, col=2)

        fig.update_layout(height = 600, width = 1200,
            scene ={'xaxis_title':r'h_{2,0}','yaxis_title':r'c_{2,0}','domain':{'x':[0,.29]},
                'aspectmode':'manual','aspectratio':dict(x=1, y=1, z=1.),'camera':dict(eye=dict(x=1.5, y=2.7, z=0.9))},
            scene2={'xaxis_title':r'h_{2,0}','yaxis_title':r'c_{2,0}','domain':{'x':[0.5,.79]},
                'aspectmode':'manual','aspectratio':dict(x=1, y=1, z=1.),'camera':dict(eye=dict(x=1.5, y=2.7, z=0.9))})
        tempFileName = d.fileName[:d.fileName.find('Us')]+d.fileName[d.fileName.find('Us') + d.fileName[:d.fileName.find('Us')].find('_')+2:]
        fig.write_html(d.rootFile + 'solutions/plots/3Dview_layeredSettling_' + tempFileName + '.html', include_mathjax="cdn")
        del fig
    singleSettlingSpeeds()
    layeredSettlingSpeeds()

def sediment_check(u_s,rootFile,N=5000,sharp=50):
    x = np.linspace(0.7,1.42,73)
    X = np.zeros((x.size,x.size))
    for i in range(x.size):
        for j in range(x.size):
            if x[i]*x[j]<=1:
                X[i,j] = min(TurbiditySim(x[i],x[j],u_s,rootFile,['d1','d2'],N=N,sharp=sharp).d1.shape[0]-1,1)
    return X

def collision_details(u_s,rootFile,N=5000,sharp=50):
    x = np.linspace(0.7,1.42,73)
    X = np.zeros((x.size,x.size))
    T = np.zeros((x.size,x.size))
    for i in range(x.size):
        for j in range(x.size):
            if x[i]*x[j]<=1:
                X[i,j] = TurbiditySim(x[i],x[j],u_s,rootFile,[],N=N,sharp=sharp).coll_loc
                T[i,j] = TurbiditySim(x[i],x[j],u_s,rootFile,[],N=N,sharp=sharp).coll_time
            else:
                X[i,j] = np.nan
                T[i,j] = np.nan
    plt.figure(figsize = [12,5])

    plt.rcParams.update({"text.usetex":True})
    figure_title = ['Collision $x$-location','Collision time']
    for i,A in enumerate([X,T]):
        plt.subplot(1,2,i+1)
        pplot = plt.pcolormesh(x,x,A,shading='nearest')
        plt.contour(x,x,A,colors = 'k',linewidths=0.75)
        plt.gca().set_aspect('equal')
        plt.colorbar(pplot)
        plt.title(figure_title[i])
        plt.xlabel('$h_{2,0}$')
        plt.ylabel('$c_{2,0}$')
    plt.tight_layout()
    plt.savefig(rootFile + 'solutions/plots/collision_details_Us%0.3f'%(u_s) + '.pdf')
    plt.show()
    plt.rcParams.update({"text.usetex":False})

def NumericalValidation(rootFile='NumericalValidation_2025Mar19/',N=20000,h_min=0.0001,NuRe=1000,CFL=0.1,sharp=200,U_s=0.0,T=40.0,apart = 5, FrSquared = 2.828, plot_=True):
    def plot_numer(X,Y,which_test,my_label,variable_y_label,par_list,x_Min,x_Max):
        plt.rcParams.update({"text.usetex":True,'font.size':16,'lines.linewidth':3,'legend.fontsize':16,'xtick.labelsize':14,'ytick.labelsize':14})
        plt.figure(figsize = [6,5])

        for x,y,par in zip(X,Y,par_list):
            idx = np.where(x>x_Min)[0]
            x = x[idx]
            y = y[idx]
            idx = np.where(x<x_Max)[0]
            x = x[idx]
            y = y[idx]

            str_label ='$%s = %i$'%(my_label,par) if isinstance(par,int) else '$%s = %f$'%(my_label,par)
            plt.plot(x,y,label = str_label)
        plt.xlabel('$x$')
        plt.ylabel(variable_y_label)
        plt.legend()

        plt.savefig(rootFile + 'solutions/plots/' + which_test + '_' + variable_y_label + '.pdf')
        plt.close()
        plt.rcParams.update({"text.usetex":False})

    def print_latex_table(var,label,M):
        print('')
        label.append(77)
        M = np.vstack((np.array(label),M))
        l = ['S','u_-','u_+','h_-','h_+']
        l.insert(0,var)
        rows,cols = M.shape
        for i in range(rows):
            string_ = '        $'+l[i]+'$'
            for j in range(cols):
                string_ += ' & %s'%('\\%') if i==0 and j == M.shape[1]-1 else ' & %0.6f '%M[i,j]
            string_ += '\\\\'
            if i==0:
                string_ += ' \\hline'
            print(string_)
        print('')
    def NumericalValidation_NuRe(rootFile='NumericalValidation_2025Mar19/',N=20000,h_min=0.0001,NuRe=1000,CFL=0.1,sharp=200,U_s=0.0):
        par_list = [250,500,1000,2000]
        par_matrix = np.zeros((5,len(par_list)+1))
        x_min_bore = 1000000
        x_max_bore = 0
        H_plot = []
        U_plot = []
        X_plot = []

        for i in range(len(par_list)):
            t, bore, hp, hm, up, um, xx,yy,zz=u_pm(subSampleBy=1,rootFile='NumericalValidation_2025Mar19/',rootFileName='',N=N,h_min=h_min,NuRe = par_list[i],CFL=0.1,sharp=200)
            x_min_bore = bore[-1] if bore[-1]<x_min_bore else x_min_bore
            x_max_bore = bore[-1] if bore[-1]>x_max_bore else x_max_bore

            for j,val in enumerate([bore[-1], um[-1], up[-1], hm[-1], hp[-1]]):
                par_matrix[j,i] = val
            x,T_vec,U = unpack_fo_real('u',rootFile='NumericalValidation_2025Mar19/',rootFileName='',N=N,h_min=h_min,NuRe = par_list[i],CFL=0.1,sharp=200,T=T)
            H = unpack_fo_real('h',rootFile='NumericalValidation_2025Mar19/',rootFileName='',N=N,h_min=h_min,NuRe = par_list[i],CFL=0.1,sharp=200,T=T)[-1]
            X_plot.append(x)
            H_plot.append(H[-1,1:])
            U_plot.append(U[-1,1:])

        plot_numer(X_plot,H_plot,'Reynolds','\\textrm{Re}','height',par_list,x_min_bore-0.5,x_max_bore+0.5);
        plot_numer(X_plot,U_plot,'Reynolds','\\textrm{Re}','velocity',par_list,x_min_bore-0.5,x_max_bore+0.5);
        for j in range(par_matrix.shape[0]):
            par_matrix[j,-1] = 100*np.abs((par_matrix[j,-2]-par_matrix[j,0])/par_matrix[j,-2])
        print_latex_table('\\Rey', par_list, par_matrix)
        return par_matrix

    def NumericalValidation_CFL(rootFile='NumericalValidation_2025Mar19/',N=20000,h_min=0.0001,NuRe=1000,CFL=0.1,sharp=200,U_s=0.0):
        par_list = [0.4,0.2,0.1,0.05]
        par_matrix = np.zeros((5,len(par_list)+1))
        x_min_bore = 1000000
        x_max_bore = 0 
        H_plot = []
        U_plot = []
        X_plot = []

        for i in range(len(par_list)):
            t, bore, hp, hm, up, um, xx,yy,zz=u_pm(subSampleBy=1,rootFile='NumericalValidation_2025Mar19/',rootFileName='',N=N,h_min=h_min,NuRe = NuRe,CFL=par_list[i],sharp=200)
            x_min_bore = bore[-1] if bore[-1]<x_min_bore else x_min_bore
            x_max_bore = bore[-1] if bore[-1]>x_max_bore else x_max_bore

            for j,val in enumerate([bore[-1], um[-1], up[-1], hm[-1], hp[-1]]):
                par_matrix[j,i] = val
            x,T_vec,U = unpack_fo_real('u',rootFile='NumericalValidation_2025Mar19/',rootFileName='',N=N,h_min=h_min,NuRe = NuRe,CFL=par_list[i],sharp=200,T=T)
            H = unpack_fo_real('h',rootFile='NumericalValidation_2025Mar19/',rootFileName='',N=N,h_min=h_min,NuRe = NuRe,CFL=par_list[i],sharp=200,T=T)[-1]
            X_plot.append(x)
            H_plot.append(H[-1,1:])
            U_plot.append(U[-1,1:])

        plot_numer(X_plot,H_plot,'CFL','\\Lambda','height',par_list,x_min_bore-0.5,x_max_bore+0.5);
        plot_numer(X_plot,U_plot,'CFL','\\Lambda','velocity',par_list,x_min_bore-0.5,x_max_bore+0.5);
        for j in range(par_matrix.shape[0]):
            par_matrix[j,-1] = 100*np.abs((par_matrix[j,-2]-par_matrix[j,0])/par_matrix[j,-2])
        print_latex_table('\\Lambda', par_list, par_matrix)
        return par_matrix
    def NumericalValidation_Sharp(rootFile='NumericalValidation_2025Mar19/',N=20000,h_min=0.0001,NuRe=1000,CFL=0.1,sharp=200,U_s=0.0):
        par_list = [50,100,200,400]
        par_matrix = np.zeros((5,len(par_list)+1))
        x_min_bore = 1000000
        x_max_bore = 0
        H_plot = []
        U_plot = []
        X_plot = []

        for i in range(len(par_list)):
            t, bore, hp, hm, up, um, xx,yy,zz=u_pm(subSampleBy=1,rootFile='NumericalValidation_2025Mar19/',rootFileName='',N=N,h_min=h_min,NuRe = NuRe ,CFL=0.1,sharp=par_list[i])
            x_min_bore = bore[-1] if bore[-1]<x_min_bore else x_min_bore
            x_max_bore = bore[-1] if bore[-1]>x_max_bore else x_max_bore

            for j,val in enumerate([bore[-1], um[-1], up[-1], hm[-1], hp[-1]]):
                par_matrix[j,i] = val
            x,T_vec,U = unpack_fo_real('u',rootFile='NumericalValidation_2025Mar19/',rootFileName='',N=N,h_min=h_min,NuRe = NuRe,CFL=0.1,sharp=par_list[i],T=T)
            H = unpack_fo_real('h',rootFile='NumericalValidation_2025Mar19/',rootFileName='',N=N,h_min=h_min,NuRe = NuRe,CFL=0.1,sharp=par_list[i],T=T)[-1]
            X_plot.append(x)
            H_plot.append(H[-1,1:])
            U_plot.append(U[-1,1:])

        plot_numer(X_plot,H_plot,'Sharp','\\sigma','height',par_list,x_min_bore-0.5,x_max_bore+0.5);
        plot_numer(X_plot,U_plot,'Sharp','\\sigma','velocity',par_list,x_min_bore-0.5,x_max_bore+0.5);
        for j in range(par_matrix.shape[0]):
            par_matrix[j,-1] = 100*np.abs((par_matrix[j,-2]-par_matrix[j,0])/par_matrix[j,-2])
        print_latex_table('\\sigma', par_list, par_matrix)
        return par_matrix

    def NumericalValidation_hmin(rootFile='NumericalValidation_2025Mar19/',N=20000,h_min=0.0001,NuRe=1000,CFL=0.1,sharp=200,U_s=0.0):
        par_list = [0.0004,0.0002,0.0001,0.00005]
        par_matrix = np.zeros((5,len(par_list)+1))
        x_min_bore = 1000000
        x_max_bore = 0
        H_plot = []
        U_plot = []
        X_plot = []

        for i in range(len(par_list)):
            t, bore, hp, hm, up, um, xx,yy,zz=u_pm(subSampleBy=1,rootFile='NumericalValidation_2025Mar19/',rootFileName='',N=N,h_min=par_list[i],NuRe = 1000,CFL=0.1,sharp=200)
            x_min_bore = bore[-1] if bore[-1]<x_min_bore else x_min_bore
            x_max_bore = bore[-1] if bore[-1]>x_max_bore else x_max_bore

            for j,val in enumerate([bore[-1], um[-1], up[-1], hm[-1], hp[-1]]):
                par_matrix[j,i] = val
            x,T_vec,U = unpack_fo_real('u',rootFile='NumericalValidation_2025Mar19/',rootFileName='',N=N,h_min=par_list[i],NuRe = NuRe,CFL=0.1,sharp=200,T=T)
            H = unpack_fo_real('h',rootFile='NumericalValidation_2025Mar19/',rootFileName='',N=N,h_min=par_list[i],NuRe = NuRe,CFL=0.1,sharp=200,T=T)[-1]
            X_plot.append(x)
            H_plot.append(H[-1,1:])
            U_plot.append(U[-1,1:])

        plot_numer(X_plot,H_plot,'hmin','h_{\\textrm{min}}','height',par_list,x_min_bore-0.5,x_max_bore+0.5);
        plot_numer(X_plot,U_plot,'hmin','h_{\\textrm{min}}','velocity',par_list,x_min_bore-0.5,x_max_bore+0.5);
        for j in range(par_matrix.shape[0]):
            par_matrix[j,-1] = 100*np.abs((par_matrix[j,-2]-par_matrix[j,0])/par_matrix[j,-2])
        print_latex_table('\\hmin', par_list, par_matrix)
        return par_matrix

    def NumericalValidation_N(rootFile='NumericalValidation_2025Mar19/',N=20000,h_min=0.0001,NuRe=1000,CFL=0.1,sharp=200,U_s=0.0):
        par_list = [5000,10000,20000,40000]
        par_matrix = np.zeros((5,len(par_list)+1))
        x_min_bore = 1000000
        x_max_bore = 0
        H_plot = []
        U_plot = []
        X_plot = []

        for i in range(len(par_list)):
            t, bore, hp, hm, up, um, xx,yy,zz=u_pm(subSampleBy=1,rootFile='NumericalValidation_2025Mar19/',rootFileName='',N=par_list[i],h_min=0.0001,NuRe = 1000,CFL=0.1,sharp=200)
            x_min_bore = bore[-1] if bore[-1]<x_min_bore else x_min_bore
            x_max_bore = bore[-1] if bore[-1]>x_max_bore else x_max_bore

            for j,val in enumerate([bore[-1], um[-1], up[-1], hm[-1], hp[-1]]):
                par_matrix[j,i] = val
            x,T_vec,U = unpack_fo_real('u',rootFile='NumericalValidation_2025Mar19/',rootFileName='',N=par_list[i],h_min=h_min,NuRe = NuRe,CFL=0.1,sharp=200,T=T)
            H = unpack_fo_real('h',rootFile='NumericalValidation_2025Mar19/',rootFileName='',N=par_list[i],h_min=h_min,NuRe = NuRe,CFL=0.1,sharp=200,T=T)[-1]
            X_plot.append(x)
            H_plot.append(H[-1,1:])
            U_plot.append(U[-1,1:])

        plot_numer(X_plot,H_plot,'SpaceDiscretization','\\Delta x','height',par_list,x_min_bore-0.5,x_max_bore+0.5);
        plot_numer(X_plot,U_plot,'SpaceDiscretization','\\Delta x','velocity',par_list,x_min_bore-0.5,x_max_bore+0.5);
        for j in range(par_matrix.shape[0]):
            par_matrix[j,-1] = 100*np.abs((par_matrix[j,-2]-par_matrix[j,0])/par_matrix[j,-2])
        print_latex_table('\\Delta x', [100/N for N in [5000, 10000, 20000, 40000]], par_matrix)
        return par_matrix
    NumericalValidation_N()
    NumericalValidation_hmin()
    NumericalValidation_NuRe()
    NumericalValidation_Sharp()
    NumericalValidation_CFL()

def article_plots(Figs=list(range(1,12))):
    article_params()
    if 1 in Figs:
        Fig1Sims = []
        for test in ((1.0,1.0),(0.7,0.7)):
            Fig1Sims.append(
                TurbiditySim(
                    *test,
                    0.0,
                    'Apr22_FinalResults/',
                    ['h','u','c1','c2','d1','d2'],
                    N=12000,
                    finalTime=12,
                    sharp=200
                )
            )
    # if 8 in Figs:
    #     MainSim = TurbiditySim(1.0,1.0,0.0,'Nov24_AsymBoxModel/',['h','u','c1','c2'],sharp = 100,N=7000)
    if 7 in Figs or 8 in Figs or 2 in Figs:
        Fig2_7_8Sims = []
        for test in ((1.0,1.0),(0.7,0.7)):
            Fig2_7_8Sims.append(
                TurbiditySim(
                    *test,
                    0.0,
                    'Apr22_FinalResults/',
                    ['h','u','c1','c2'],
                    N=6000,
                    finalTime=6,
                    sharp=200
                )
            )
    if 6 in Figs:
        MainDep = DepositionAnalysis(0.01,'SedimentationInitialConditionTest_2025Jun7/')
    if 1 in Figs:
        # Figure 1,  Results - solution profile
        for sim in Fig1Sims:
            sim.plot_profile_results(['h','u','c1','c2'])
        # Make it look not dumb
        # needs to go somewhere else, but not sure where
        # Make a matching MP4 to go with supplemental materials 
    if 2 in Figs:
        # Figure 2,  Example solution for numerics discussion
        Fig2_7_8Sims[1].num_val_schematic(6,show=False)
        # Remove shading
        # Add x_i to left panel
        # Make arrows thinner
        # leave h_r/c_r 
        # Keep all axes equal
        # vertically stacked, with no space between figures (since they have a shared axis), with tick marks on each axis (but no labels) 
    if 3 in Figs:
        pass
        # Figure 3,  Numerical validation for spatial resolution
    if 4 in Figs:
        pass
        # Figure 4,  Numerical validation for Reynolds number
    if 5 in Figs:
        pass
        # Figure 5,  Sediment deposition - example solutions
        # Matching vertical axes
        # vertically stacked
        # Separate figures, since each one is a different run
    if 6 in Figs:
        # Figure 6,  Sediment deposition - Encroachment mass colormap
        MainDep.myPcolor('encroachment_mass','',save=True)
        # add a,b,c,d at correct locations aligning with figure 6
        # change labels to c_r and h_r
        # only 1 significant digit on the colorbar
        # OLD Figure,  Sediment deposition - Encroachment mass 3D view with planar approximation
        # This figure has gone away,
        # but we want find a "best fit" parameter to report for planes. 
    if 7 in Figs:
        for sim in Fig2_7_8Sims:
            sim.spacetime(xlim=[-5,5])
        # hTwo0.70_cTwo0.70_5apart_N6000_CFL0.100_T6.0_NuRe1000_Us0.000_hmin0.00010_sharp200
        # Figure 7,  Results - Space time plot
        # Add test case matching figure 4
    if 8 in Figs:
        # Figure 8,  Box model - schematic
        # MainSim.box_model_schematic(6,show=False)
        Fig2_7_8Sims[0].box_model_schematic(6,show=False)
        # Remove top two panels, they are redundant with figure 1
        # Switch to be avg h-/+ for third panel
    if 9 in Figs:
        # Figure 9, Box model - results - asymmetric currents with no shape factor
        # Change velocity ymin to be 0
        # make 0 lower bound be tight on all plots. 
        # kill the horizontal label on top subplot. 
        Box_SWE_Asym()
    if 10 in Figs:
        # Figure 10, Box model - results - asymmetric currents with shape factor = 0.9
        # Change velocity ymin to be 0
        Box_SWE_Asym(shape_factor=0.9)
    if 11 in Figs:
        # Figure 11
        # Change velocity ymin to be 0
        Box_SWE_Settling()
