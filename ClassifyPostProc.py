import numpy as np
import matplotlib.pyplot as plt
import sys
if float(str(sys.version_info[1]) + '.' + str(sys.version_info[2])) <= 9.1:
    from matplotlib import cm
else:
    from matplotlib import colormaps as cm
import matplotlib.animation as ani
from celluloid import Camera
from os import getcwd
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
    plt.rcParams.update({"text.usetex":True,"figure.figsize":[3.5,3],"font.family": "serif","font.sans-serif": ["Helvetica"],'font.size':10,'lines.linewidth':2,'legend.fontsize':9,'xtick.labelsize':8,'ytick.labelsize':8})
def viewing_params():
    plt.rcParams.update({"text.usetex":False,"figure.figsize":[12,9],"font.family": "serif","font.sans-serif": ["Helvetica"],'font.size':22,'lines.linewidth':3,'legend.fontsize':20,'xtick.labelsize':20,'ytick.labelsize':20})
viewing_params()
tabcolors = ('tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan')
mpllinestyles = ('solid','dashed','dotted','dashdot',(0, (3, 5, 1, 5, 1, 5)))
mpllinestyles = ('solid','dashed','dotted','dashdot')
mplmarkers = ('o','v','^','<','>','s','*','D','P','X')

variable_dict = {'c':'concentration', 'c1':'left concentration', 'c2':'right concentration', 'u':'velocity', 'q':'q, conserved velocity', 'phi':'phi, conserved concentration', 'h':'height','h_latex':'$h(x,t)$','u_latex':'$u(x,t)$','c1_latex':'$c_1(x,t)$','c2_latex':'$c_2(x,t)$'}
char_to_cons = {'u':'q','c1':'phi1','c2':'phi2'}
char_vars = ['u','c1','c2']

testFileName = 'SedimentationInitialConditionTest_2025Jun7/'
class TurbiditySim:

    def __init__(self, hR0, cR0, U_s, rootFile,VARS, N = 5000, NuRe = 1000, finalTime = 40., h_min = 0.0001, CFL = 0.1, sharp = 50, apart = 5., FrSquared = 1., hL0 = 1.0, cL0 = 1.0, NuPe = None, subFile = 'sims/'):
        self.hR0 = hR0
        self.cR0 = cR0
        self.U_s = U_s
        self.rootFile = rootFile
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
        
        self.fileName = "hOne%0.2f_hTwo%0.2f_cOne%0.2f_cTwo%0.2f_%iapart_N%i_CFL%0.3f_T%0.1f_NuRe%i_NuPe%i_FrFr%0.3f_Us%0.3f_hmin%0.5f_sharp%i"%(self.hL0,self.hR0,self.cL0,self.cR0,self.apart,self.N,self.CFL,self.finalTime,self.NuRe,self.NuPe,self.FrSquared,self.U_s,self.h_min,self.sharp)
        
        self.unpack(VARS)
        self.dt = self.T[1]-self.T[0]
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

    def makeMP4(self, varList = ['h','u','c1','c2'],tMax=1000.,show_legend = False, xlim = None, ylim = None,framerate = 30.):#xmax = None, xmin = None,ymax = None, ymin = ):

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
        VidPath = getcwd()[:getcwd().find('/D')+1] + "Documents/MercedResearch/WithFrancois/Turbidity/FinVol/" + self.rootFile + "solutions/videos/"
        animat.save(VidPath + self.fileName.replace('.','_') + '_tMax%i'%tMax + '.mp4', writer = Writ)
        plt.close()

    def plot_time(self,var,desired_time,xlim = None):
        index = np.argmin(np.abs(self.T-desired_time))
        x_min_idx, x_max_idx = (np.argmin(np.abs(self.x-xlim[0])), np.argmin(np.abs(self.x-xlim[1]))) if xlim else (None, None)

        plt.plot(self.x[x_min_idx:x_max_idx],getattr(self,var)[index,x_min_idx:x_max_idx],label = '$t=%0.1f$'%self.T[index])

        plt.xlabel('$x$')
        plt.ylabel(variable_dict[var])

    def plot_times(self,var,times,xlim=None,wl = True):
        for t in times:
            self.plot_time(var,t)
        plt.xlim(xlim)
        if wl: plt.legend()

    def plot_examples(self,var,times=[0,4,8,12]):
        article_params()
        plt.figure(figsize = [8,4])
        for i,v in enumerate(var):
            plt.subplot(len(var),1,i+1)
            self.plot_times(v,times,xlim=[-10,10],wl=False)
            plt.ylabel(variable_dict[v + '_latex'])
            if i < len(var)-1: plt.gca().set_xticks([])
        
        plt.subplot(len(var),1,1)
        plt.legend(['$t=%0.1f$'%t for t in times],ncol=len(times),loc='upper center', bbox_to_anchor=(0.5, 1.6))
        plt.subplot(len(var),1,len(var))
        plt.xlabel('$x$')
        plt.savefig(self.rootFile + 'solutions/plots/solution_example_' + self.fileName + '.png', bbox_inches='tight',dpi=400)
        

    def deposition_details(self):
        dx = self.x[1]-self.x[0]

        self.xr = self.x[self.coll_idx+1:int(self.N*0.8)]
        self.l2r = self.d1[-1,self.coll_idx+1:int(self.N*0.8)]

        self.xl = self.x[int(self.N*0.2):self.coll_idx+1]
        self.r2l = self.d2[-1,int(self.N*0.2):self.coll_idx+1]

        two_sided_intrusion = np.hstack([self.r2l,self.l2r])
        two_sided_x = np.hstack([self.xl,self.xr])
        
        l2r_mass = np.sum(self.l2r)*dx
        r2l_mass = np.sum(self.r2l)*dx
 
        #self.intrusion_mass = l2r_mass if l2r_mass >= r2l_mass else r2l_mass
        #self.COM_x = np.sum(self.xr*self.l2r)*dx/l2r_mass if l2r_mass >= r2l_mass else np.sum(self.xl*self.r2l)*dx/r2l_mass
        self.intrusion_mass = l2r_mass - r2l_mass
        self.COM_x = np.sum(two_sided_x*two_sided_intrusion)*dx/(l2r_mass + r2l_mass)

    def plot_deposit(self, LS = 'solid'):
        plt.plot(self.x,self.d1[-1,:],label='$d_1(x), U_s = %0.3f$'%self.U_s,color = 'tab:blue', linestyle = LS)
        plt.plot(self.x,self.d2[-1,:],label='$d_2(x), U_s = %0.3f$'%self.U_s,color = 'tab:orange', linestyle = LS)
        plt.xlim([-20,20])
        plt.title('$h_{2,0}$ = %0.2f, $c_{2,0}$ = %0.2f'%(self.hR0,self.cR0))
        plt.xlabel('$x$')

    def plot_intrusion(self,wantLegend=True, want_y_label=True):
        try:
            self.intrustion_mass
        except AttributeError:
            self.deposition_details()
        plt.plot(self.xl,self.r2l,label = 'right-to-left: $d_2(x), x>x_c$')
        plt.plot(self.xr,self.l2r,label = 'left-to-right: $d_1(x), x<x_c$')
        #plt.scatter(self.COM_x,self.intrusion_mass,)
        plt.plot([self.coll_loc]*2,[0,max(np.max(self.l2r),np.max(self.r2l))],color = 'k',linestyle = 'dashed',label = 'collision location $(x_c)$: %0.2f'%self.coll_loc,linewidth = 2)
        plt.plot([self.COM_x]*2,[0,max(np.max(self.l2r),np.max(self.r2l))],color = 'red',linestyle = 'dashed',label = 'COM $x$-coordinate: %0.2f'%self.COM_x)
        try: 
            xMin = self.xl[np.argwhere(self.r2l>0.01*np.max(self.r2l))[0][0]]
            xMax = self.xr[np.argwhere(self.l2r>0.01*np.max(self.l2r))[-1][0]]
            plt.xlim([xMin,xMax])
        except IndexError:
            print('Adaptive xlim did not work, setting bounds at [-5,5]')
            plt.xlim([-5,5])
        plt.title('$h_{2,0}$ = %0.2f, $c_{2,0}$ = %0.2f \n $\int_{x_c}^\infty d_1(x) dx - \int_{-\infty}^{x_c} d_2(x) dx $= %0.2f'%(self.hR0,self.cR0,self.intrusion_mass))
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
        try: 
            C=np.loadtxt(self.rootFile + 'solutions/postData/post_collision_bore_data_'+ self.fileName + '.csv',delimiter=',')
            t_post = C[:,0]
            bore = C[:,1]
            h_plus,   h_minus   = C[:,2], C[:,3]
            u_plus,   u_minus   = C[:,4], C[:,5]
            h_plus_avg,   h_minus_avg   = C[:,6], C[:,7]
            #phi_plus, phi_minus = C[:,6], C[:,7]
            front = C[:,8]
        except OSError: 
            print('cannot open, so I will post process the data MYSELF!')
            t_post = []
            u_plus, u_minus = [],[]
            h_plus, h_minus = [],[]
            h_plus_avg, h_minus_avg = [],[]
            bore, front = [],[]
            THRESH = []

            dt = self.T[1]-self.T[0]
            post_collision = False # Flag to determine if the currents have collided yet. 
            h_max = 0
            bore_index=0
            t_count = 0
            x_ = self.x[int(self.N/2):]
            for t,u_fake,h_fake in zip(self.T,self.u[:,int(self.N/2):],self.h[:,int(self.N/2):]):
                u_ = deepcopy(u_fake)
                h_ = deepcopy(h_fake)
                window_size = 50
                if (not post_collision) and h_[0]>2*self.h_min and np.max(h_)<h_max and x_[np.argmax(h_)]<2 and t>t_start:
                    post_collision = True
                front_index = np.argwhere(h_>2*self.h_min)[-1][0]
                if post_collision:
                    threshold = (h_[int((front_index+bore_index)/2)] + h_[int(bore_index/2)])/2
                    THRESH.append(threshold)
                    u_max = np.max(u_)
                    if bore_index:
                         h_[:max(bore_index-2*window_size,0)]=np.max(h_)
                         h_[min(bore_index+2*window_size,len(h_)):]=self.h_min
                    bore_index = np.argwhere(h_>threshold)[-1][0]
                    u_bore = u_[max(bore_index-window_size,0):bore_index+window_size]
                    h_bore = h_[max(bore_index-window_size,0):bore_index+window_size]
                    x_bore = x_[max(bore_index-window_size,0):bore_index+window_size]
                    u_mbi, u_pbi = np.argmax(u_bore), np.argmin(u_bore)
                    #if u_mbi == 0: continue

                    bore_loc = (x_[bore_index]-x_[bore_index+1])*(threshold-h_[bore_index+1])/(h_[bore_index]-h_[bore_index+1])+x_[bore_index+1]
                    if bore_loc < 0.2: continue
                    #if plot and (not int(t)%2) and np.abs(int(t)-t)<dt/2:
                    if plot and t<2.5:
                        plt.subplot(211)
                        p1 = plt.plot(x_bore,h_bore,label='t=%0.2f'%t)[0]
                        h_boxed = np.array([h_bore[u_mbi] if x_b<bore_loc else h_bore[u_pbi] for x_b in x_bore])
                        plt.plot(x_bore,h_boxed,linestyle = 'dashed',color = p1.get_color())

                        plt.subplot(212)
                        #plt.plot(x_bore,u_bore,color = p1.get_color())
                        plt.plot(x_bore,u_bore)
                     
                    t_post.append(t-self.coll_time)
                    u_minus.append(u_bore[u_mbi])
                    u_plus.append(u_bore[u_pbi])
                    h_minus.append(h_bore[u_mbi])
                    h_plus.append(h_bore[u_pbi])

                    front_loc = x_[front_index]

                    h_minus_avg.append(np.sum(h_[:bore_index])*self.dx/bore_loc)
                    h_plus_avg.append(np.sum(h_[bore_index:front_index])*self.dx/(front_loc-bore_loc))
                    bore.append(bore_loc)
                    #front_loc = (x_[front_index]-x_[front_index+1])*(2*h_min-h_[front_index+1])/(h_[front_index]-h_[front_index+1])+x_[front_index+1]
                    front.append(front_loc)
                if plot: 
                    plt.subplot(212)
                    plt.xlabel('x')
                    plt.ylabel('velocity')

                    plt.subplot(211)
                    plt.xlabel('x')
                    plt.ylabel('height')

                    plt.legend()
                h_max = np.max(h_)
    
            array_to_save=np.array((np.array(t_post),np.array(bore),np.array(h_plus),np.array(h_minus),np.array(u_plus),np.array(u_minus),np.array(h_plus_avg),np.array(h_minus_avg),np.array(front))).T
            if save: np.savetxt(self.rootFile + 'solutions/postData/post_collision_bore_data_' + self.fileName + '.csv',array_to_save,delimiter=',')
        def subsample(x,subSampleBy = subSampleBy):
            x = np.array(x)
            return x[range(0,x.shape[0],subSampleBy)]
        self.t_post = subsample(t_post)
        self.bore = subsample(bore)
        self.hP_data, self.hM_data = subsample(h_plus),subsample(h_minus)
        self.hP_data_avg, self.hM_data_avg = subsample(h_plus_avg),subsample(h_minus_avg)
        self.uP_data, self.uM_data = subsample(u_plus),subsample(u_minus)
        self.front_data = subsample(front)

    def RH_model(self, t0=0, b0=0, val=None, dt=0.001, f_vel=1.63307, x0=2.5,V=1,final = np.inf,hpf=1,upf=1,hmf=1):
        dSdt    = lambda hP,hM,uP   : uP + np.sqrt((1/2)*(hM/hP)*(hP+hM)) 
        h_plus  = lambda V,xN,xI    : V/(2*(xN-xI))*hpf
        u_plus  = lambda uN,xN,xI,S : uN/(xN-xI)*(S-xI)*upf
        h_minus = lambda V,hP,xN,S  : (V-hP*(xN-S))/S*hmf

        t,b = t0,b0
        T, B, F, B_v = [], [], [], []
        hM, hP, uP = [], [], []
        while t<=min(final,self.T[-1]-self.coll_time):
            B.append(b)
            T.append(t)

            f = 2*x0 + f_vel*(t)
            F.append(f)
            hP_ = h_plus(V,f,x0)
            uP_ = u_plus(f_vel,f,x0,b)
            hM_ = h_minus(V,hP_,f,b)

            c = dSdt(hP_,hM_,uP_)

            b = b + dt*c

            B_v.append(c)
            hP.append(hP_)
            hM.append(hM_)
            uP.append(uP_)

            t+=dt

        self.hP= np.array(hP)
        self.hM= np.array(hM)
        self.uP= np.array(uP)

        self.F = np.array(F)
        self.B = np.array(B)
        self.B_vel = np.array(B_v)
        self.t_num = np.array(T)

    def RH_plots(self, show=True):
        plt.figure(figsize = [7.5,7.5]) 

        plt.subplot(221)
        plt.plot(self.t_post,self.hP_data,label = 'data')
        plt.plot(self.t_num,self.hP,label = 'model')
        plt.xlabel('time')
        plt.ylabel('$h^+$')
        plt.legend()
        plt.subplot(222)
        plt.plot(self.t_post,self.hM_data,label = 'data')
        plt.plot(self.t_num,self.hM,label = 'model')
        plt.xlabel('time')
        plt.ylabel('$h^-$')
        plt.legend()
        plt.subplot(223)
        plt.plot(self.t_post,self.uP_data,label = 'data')
        plt.plot(self.t_num,self.uP,label = 'model')
        plt.xlabel('time')
        plt.ylabel('$u^+$')
        plt.legend()
        plt.subplot(224)
        plt.plot(self.t_post,self.vel(self.bore),label = 'data')
        plt.plot(self.t_num,self.B_vel,label = 'model')
        plt.xlabel('time')
        plt.ylabel("$\\frac{dx_b}{dt}$")
        plt.legend()

        plt.subplots_adjust(left = 0.08,right = 0.99, top = 0.99, bottom = 0.07, hspace = 0.15, wspace = 0.2)

        if show: plt.show()

    def vel(self,x,sigma=2):
        dt = self.t_post[1]-self.t_post[0]
        v = (x[2:]/2-x[:-2]/2)/dt
        v0 = (-3/2*x[0]+2*x[1]-1/2*x[2])/dt
        vN = (1/2*x[-3]-2*x[-2]+3/2*x[-1])/dt
        print('This velocity data has been filtered with a Gaussian filter with Sigma = %i'%sigma)
        return gaussian_filter1d(np.hstack((v0,v,vN)),sigma)

    def box_model_schematic(self,time,show=True):
        self.bore_data()
        idx = np.argmin(np.abs(self.t_post - (time-self.coll_time)))
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
        self.plot_time('h', time, xlim=[0,x_upper_bound])
        plt.plot([bore_pos]*2,[0, hm],color = 'k',linestyle = 'dashed',linewidth = 2)
        plt.plot([0, bore_pos],2*[hm],color = 'k',linestyle = 'dashed',linewidth = 2)
        plt.plot([0, front_pos],2*[0],color = 'k',linestyle = 'dashed',linewidth = 2)
        plt.plot([bore_pos, front_pos],2*[hp],color = 'k',linestyle = 'dashed',linewidth = 2)
        plt.plot([front_pos]*2,[0, hp],color = 'k',linestyle = 'dashed',linewidth = 2)

        plt.annotate('$h_-$',xy=(bore_pos+0.02*x_upper_bound,hm),xytext=(bore_pos + 0.15*x_upper_bound,hm), horizontalalignment='center', verticalalignment = 'center',arrowprops=dict(facecolor='black',width = 1, headwidth = 6,headlength=8))
        plt.annotate('$h_+$',xy=(bore_pos-0.02*x_upper_bound,hp),xytext=(bore_pos - 0.15*x_upper_bound,hp), horizontalalignment='center',verticalalignment = 'center',arrowprops=dict(facecolor='black',width = 1, headwidth = 6,headlength=8))
        plt.annotate('$x_b$',xy=(bore_pos,plt.gca().get_ylim()[0]),xytext=(bore_pos,plt.gca().get_ylim()[0]-0.25*(plt.gca().get_ylim()[1]-plt.gca().get_ylim()[0])), horizontalalignment='center',verticalalignment = 'top',arrowprops=dict(facecolor='black',width = 1, headwidth = 6,headlength=8))
        plt.annotate('$x_N$',xy=(front_pos,plt.gca().get_ylim()[0]),xytext=(front_pos,plt.gca().get_ylim()[0]-0.25*(plt.gca().get_ylim()[1]-plt.gca().get_ylim()[0])), horizontalalignment='center',verticalalignment = 'top',arrowprops=dict(facecolor='black',width = 1, headwidth = 6,headlength=8))
        plt.gca().set_xticks([0,5,10])
         

        plt.subplot(122)
        self.plot_time('u', time, xlim=[0,x_upper_bound])
        plt.xlabel('$x$')
        plt.annotate('$u_+$',xy=(bore_pos+0.02*x_upper_bound,up),xytext=(bore_pos + 0.15*x_upper_bound,up), horizontalalignment='center', verticalalignment = 'center',arrowprops=dict(facecolor='black',width = 1, headwidth = 6,headlength=8))
        plt.annotate('$u_-$',xy=(bore_pos-0.02*x_upper_bound,um),xytext=(bore_pos - 0.15*x_upper_bound,um), horizontalalignment='center',verticalalignment = 'center',arrowprops=dict(facecolor='black',width = 1, headwidth = 6,headlength=8))
        plt.annotate('$x_b$',xy=(bore_pos,plt.gca().get_ylim()[0]),xytext=(bore_pos,plt.gca().get_ylim()[0]-0.25*(plt.gca().get_ylim()[1]-plt.gca().get_ylim()[0])), horizontalalignment='center',verticalalignment = 'top',arrowprops=dict(facecolor='black',width = 1, headwidth = 6,headlength=8))
        plt.gca().set_xticks([0,5,10])
        plt.annotate('$x_N$',xy=(front_pos,plt.gca().get_ylim()[0]),xytext=(front_pos,plt.gca().get_ylim()[0]-0.25*(plt.gca().get_ylim()[1]-plt.gca().get_ylim()[0])), horizontalalignment='center',verticalalignment = 'top',arrowprops=dict(facecolor='black',width = 1, headwidth = 6,headlength=8))

        plt.subplots_adjust(left = 0.07,right = 0.99, bottom = 0.3, top = 0.98,wspace = 0.15)
        plt.savefig(self.rootFile + 'solutions/plots/' + 'RhModel_' + self.fileName + '.png',dpi = 1200)
        plt.savefig(self.rootFile + 'solutions/plots/' + 'RhModel_' + self.fileName + '.pdf')
        if show: plt.show()

    def spacetime(self,xlim=[None,None],tmax = None,show=False):
        from parmat import cm_data
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('mypar', cm_data, N=256)
    
        article_params()
        plt.figure(figsize=[3.5,2.75])
        x_min_idx = np.argmin(np.abs(self.x-xlim[0])) if xlim[0] else 0
        x_max_idx = np.argmin(np.abs(self.x-xlim[1])) if xlim[1] else self.N
        t_idx     = np.argmin(np.abs(self.T-tmax))    if tmax    else len(self.T)
        fig = plt.pcolormesh(self.x[x_min_idx:x_max_idx],self.T[:t_idx],self.h[:t_idx,x_min_idx:x_max_idx],shading = 'gouraud',cmap=cmap)
        plt.colorbar(fig)
        plt.xlabel('$x$')
        plt.ylabel('$t$')

        plt.subplots_adjust(left = 0.11,right = 1, top = 0.97, bottom = 0.15)
        plt.savefig(self.rootFile + 'solutions/plots/' + 'SpaceTime_' + self.fileName + '.png',dpi = 1200)
        plt.savefig(self.rootFile + 'solutions/plots/' + 'SpaceTime_' + self.fileName + '.pdf')
        if show: plt.show()

    def front_vel_plot(self, show=True):
        plt.figure(figsize=[3.5,2])
        article_params()
        vel = np.max(self.u[:,np.argwhere(self.x>0)],1)
        plt.plot(self.T,vel)
        plt.axvspan(self.coll_time,self.T[-1],color = 'gray', alpha = 0.3)
        plt.plot(self.T[self.T>self.coll_time], self.T[self.T>self.coll_time]*0+np.average(vel[self.T>self.coll_time]), linestyle = 'dashed',color='k')
        plt.text(3*self.coll_time, 0.98*np.average(vel[self.T>self.coll_time]), '$u_N\\approx%0.2f$'%np.average(vel[self.T>self.coll_time]), verticalalignment='top', horizontalalignment='left')
    
        plt.xlabel('time, $t$')
        plt.ylabel('nose velocity, $\\frac{dx_N}{dt}$')
        plt.text((self.T[-1]+self.T[0])/2, (vel[-1]+vel[0])/2, 'post collision', verticalalignment='center', horizontalalignment='center')
    
        plt.xlim([None,self.T[-1]])
        plt.subplots_adjust(left = 0.16,bottom = 0.22, top = 0.98, right = 0.98)
        plt.savefig(self.rootFile + 'solutions/plots/' + 'FrontVel_' + self.fileName + '.png',dpi = 1200)
        plt.savefig(self.rootFile + 'solutions/plots/' + 'FrontVel_' + self.fileName + '.pdf')
        if show: plt.show()



#t = TurbiditySim(1.0,1.0,0.0,'Jul16_BoxModel/',['h','u'],N = 20000, sharp = 200)

h_plus  = lambda V,xN,xI    : V/(2*(xN-xI))
u_plus  = lambda uN,xN,xI,S : uN/(xN-xI)*(S-xI)
h_minus = lambda V,hP,xN,S  : (V-hP*(xN-S))/S
dSdt = lambda hP,hM,uP: uP + np.sqrt((1/2)*(hM/hP)*(hP+hM)) 
def test_plots(T):
    #T = TurbiditySim(1.0,1.0,0.0,'Jul16_BoxModel/',['h','u','c1','c2'],N=20000,sharp = 200)
    T.RH_model()
    T.RH_model_OLD()
    
    plt.subplot(221)
    plt.plot(T.t_post,T.hP_data,label = 'data')
    plt.plot(T.t_post,T.hP_model,label = 'model')
    plt.xlabel('time')
    plt.ylabel('$h^+$')
    plt.legend()
    plt.subplot(222)
    plt.plot(T.t_post,T.hM_data,label = 'data')
    plt.plot(T.t_post,T.hM_model,label = 'model')
    plt.xlabel('time')
    plt.ylabel('$h^-$')
    plt.legend()
    plt.subplot(223)
    plt.plot(T.t_post,T.uP_data,label = 'data')
    plt.plot(T.t_post,T.uP_model,label = 'model')
    plt.xlabel('time')
    plt.ylabel('$u^+$')
    plt.legend()
    plt.subplot(224)
    plt.plot(T.t_post,T.vel(T.bore),label = 'data')
    plt.plot(T.t_post,T.vel(T.bore_model),label = 'model')
    plt.xlabel('time')
    plt.ylabel("$S'(t)$")
    plt.legend()
 
    plt.tight_layout()

    plt.show()

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
    for i,j in enumerate(leg.legendHandles):
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
        for i,j in enumerate(leg.legendHandles):
            j.set_color(colors[i])
            j.set_linestyle(linestyles[i])

        plt.subplot(1,3,1)
        plt.ylabel('deposit')
        plt.subplots_adjust(left = 0.07,right = 0.99, bottom = 0.2, top = 0.85,wspace = 0.18)
        plt.savefig(tt.rootFile + 'solutions/plots/deposit_examples' + tt.fileName + '.png', bbox_inches='tight',dpi=400)
    def intrusion_only(us):
        article_params()
        plt.figure(figsize=[8,3])
        subplot_counter = 0
        wyl = True # want y legend? 
        for h0,c0 in zip([1.0,0.7],[1.0,1.0]):
            subplot_counter+=1
            plt.subplot(1,2,subplot_counter)
            tt = TurbiditySim(h0,c0,us,testFileName,['d1','d2'])
            tt.plot_intrusion(wantLegend=False, want_y_label=wyl)
            wyl = False # want y legend? 

        #plt.subplot(1,2,2)
        #legendlist = ['$U_s = %0.3f$'%us for us in US] + ['$d_1(x)$','$d_2(x)$'] 
        #colors = ['k' for i in range(len(US))] + list(tabcolors[:2])
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        #linestyles = list(mpllinestyles[:3]) + ['solid' for i in range(2)]
        #for i,j in enumerate(leg.legendHandles):
        #    j.set_color(colors[i])
        #    j.set_linestyle(linestyles[i])

        plt.subplots_adjust(left = 0.07,right = 0.75, bottom = 0.2, top = 0.85,wspace = 0.18)
        plt.savefig(tt.rootFile + 'solutions/plots/intrusionOnly_versus_IC_' + tt.fileName + '.png', bbox_inches='tight',dpi=400)
    full_deposit()
    intrusion_only(0.005)
    intrusion_only(0.01)
    intrusion_only(0.015)

class DepositionAnalysis:
    def __init__(self, U_s, rootFile, H2=np.linspace(0.7,1.42,73), C2=np.linspace(0.7,1.42,73), N = 5000, NuRe = 1000, finalTime = 40., h_min = 0.0001, CFL = 0.1, sharp = 50, apart = 5., FrSquared = 1., hL0 = 1.0, cL0 = 1.0, NuPe = None, subFile = 'solutions/postData/'):
        self.H2 = H2
        self.C2 = C2
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
            self.intrusion_mass = np.loadtxt(self.rootFile + self.subFile + 'intrMass_' + self.fileName + '.csv', delimiter = ',')
            self.COM_x = np.loadtxt(self.rootFile + self.subFile + 'COMx_' + self.fileName + '.csv', delimiter = ',')
        except OSError:
            print('Cannot find processed sedimentation data. Post processing now.') 
            start = time.time()
            self.intrusion_mass = np.zeros([C2.shape[0],H2.shape[0]])
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
                            self.intrusion_mass[j,i] = temp_sim.intrusion_mass
                            self.COM_x[j,i] = temp_sim.COM_x
                            self.deposited_mass[j,i] = (np.sum(temp_sim.d1)+np.sum(temp_sim.d2))*(temp_sim.x[1]-temp_sim.x[0])/(temp_sim.hL0*temp_sim.cL0 + h2*c2)
                            self.coll_time[j,i] = temp_sim.coll_time
                            self.coll_loc[j,i] = temp_sim.coll_loc
                        else:
                            print('h2 = %0.2f, c2 = %0.2f, U_s = %0.3f simulation did not finish. Final deposit data DNE'%(h2,c2,self.U_s))
                            self.intrusion_mass[j,i] = np.nan
                            self.coll_time[j,i] = np.nan
                            self.coll_loc[j,i] = np.nan
                            self.COM_x[j,i] = np.nan
                    else:
                        self.intrusion_mass[j,i] = np.nan
                        self.COM_x[j,i] = np.nan
                        self.coll_time[j,i] = np.nan
                        self.coll_loc[j,i] = np.nan
            np.savetxt(self.rootFile + self.subFile + 'intrMass_' + self.fileName + '.csv', self.intrusion_mass, delimiter = ',')
            np.savetxt(self.rootFile + self.subFile + 'COMx_' + self.fileName + '.csv', self.COM_x, delimiter = ',')
            print('%0.2f seconds'%(time.time()-start))

    def myPcolor(self,attr,plotTitle,streamlines=True):
        pplot = plt.pcolormesh(self.H2,self.C2,getattr(self,attr),shading = 'gouraud')
        plt.contour(self.H2,self.C2,getattr(self,attr),colors='k',levels=[0])
        plt.colorbar(pplot)
        plt.gca().set_aspect('equal')
        plt.title(plotTitle)
        plt.xlabel('$h_{2,0}$')
        plt.ylabel('$c_{2,0}$')

    def plot_dimensional_analysis(self):
        article_params()
        plt.figure(figsize=[7,2.75])
        plt.subplot(121)
        self.myPcolor('intrusion_mass','Intrusion Mass')
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

    def my3Dplot(self):
        H_mesh,C_mesh = np.meshgrid(self.H2,self.C2)

        intrusion_linear = self.linear_appr(H_mesh,C_mesh,self.intrusion_mass)
        intrusion_quadratic = self.quadratic_appr(H_mesh,C_mesh,self.intrusion_mass)

        fig = make_subplots(rows=1, cols=2, 
            specs=[[{'type':'surface'}]*2], 
            subplot_titles=(r'$\text{Intrusion Mass}$', r'$\text{COM }x\text{-coordinate}$'),
            horizontal_spacing=0.15)

        fig.add_trace(go.Surface(z=self.intrusion_mass, x=H_mesh, y=C_mesh, colorscale='viridis',colorbar={'x':0.43,'title':'data'}), row=1, col=1)
        fig.add_trace(go.Surface(z=intrusion_linear, x=H_mesh, y=C_mesh, colorscale='Plotly3', colorbar={'x':0.51, 'title':'Linear approximation'}), row=1, col=1)

        fig.add_trace(go.Surface(z=self.COM_x, x=H_mesh, y=C_mesh, colorscale='plasma'), row=1, col=2)

        fig.update_layout(title_text=r'$\text{Settling speed}: U_s=%0.3f$'%self.U_s,
            height = 600, width = 1200,
            scene ={'xaxis_title':r"h_{2,0}",'yaxis_title':r"c_{2,0}",
                'aspectmode':'manual','aspectratio':dict(x=1, y=1, z=1.),'camera':dict(eye=dict(x=1.1, y=1.98, z=0.66))},
            scene2={'xaxis_title':r'h_{2,0}','yaxis_title':r'c_{2,0}',
                'aspectmode':'manual','aspectratio':dict(x=1, y=1, z=1.),'camera':dict(eye=dict(x=1.1, y=1.98, z=0.66))})
        fig.write_html(self.rootFile + 'solutions/plots/3Dview_' + self.fileName +'.html', include_mathjax="cdn")
        del fig

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
            subplot_titles=(r'$\text{Intrusion Mass}$', r'$\text{COM }x\text{-coordinate}$'))
  
        settlingSpeedCM = ['Blugrn','Plotly3','Burg']
         
        for i,u in enumerate(US):
            d = DepositionAnalysis(u,'SedimentationInitialConditionTest_2025Jun7/')
            H_mesh,C_mesh = np.meshgrid(d.H2,d.C2)
             

            fig.add_trace(go.Surface(z=d.intrusion_mass, x=H_mesh, y=C_mesh, 
                colorscale=settlingSpeedCM[i],
                colorbar={'x':0.29+0.07*i,'title':{'text':'$U_s=%0.3f$'%(u),'side':'top'},'len':0.75,'thickness':20}), 
                row=1, col=1)
            fig.add_trace(go.Surface(z=d.COM_x, x=H_mesh, y=C_mesh, 
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



















#def plotTimeT(x,u,t,time):
#    i = np.abs(t-time).argmin()
#    plt.plot(x,u[i,:],label = '$t = %0.1f$'%time)
#
#def plotTimeT_xRange(x,u,t,time,xmin,xmax):
#    i = np.abs(t-time).argmin()
#    ix = np.array([x>3,x<4]).all(axis=0)
#    plt.plot(x,u[i,:],label = 't = %0.1f'%time)
#    
#def plotMultTimes(rootFile = 'ShallowWaterSimulations/' ,rootFileName = 'BoxIC_NumericalValidation_',N=4000,T=20.0,h_min=0.001,NuRe=250,NuPe=250,FrSquared=2,U_s=0.01):
#    details = buildFileName(N, T, h_min, NuRe, NuPe, FrSquared, U_s)
#    SWEFiles = [rootFileName + s + details for s in ['h','u','c']]
#    SWELegend = ['height','velocity','concentration']
#    subplotcounter = 0
#    fig = plt.figure(figsize = [20,6])
#    for filename,leg in zip(SWEFiles,SWELegend):
#        subplotcounter+=1
#        x,t,u = unpack(rootFile + filename)
#        plt.subplot(1,3,subplotcounter)
#        for time in [0,3,6,9,12,15,18]:
#            plotTimeT(x,u,t,time)
#        plt.legend()
#        plt.ylabel(leg)
#        plt.xlabel('x')
#    fig.suptitle('N = %i, h_min = %0.03f, NuRe = NuPe = %i, FrSquared = %i, U_x = %0.3f'%(N,h_min,NuRe, FrSquared, U_s))
#    filename = rootFile + 'solutions/plots/' + rootFileName + details
#    plt.savefig(filename.replace('.','_') + '.pdf')
#    plt.close()
#    
#def find_zero_center_out(x,y):
#    N = int(len(x)/2)
#    i = 1
#    while y[N-i]*y[N+i]>0:
#        i+=1
#    if y[N+i]*y[N+i-1]<0:
#       lo,up = N+i-1,N+i
#    elif y[N-i]*y[N-i+1]<0:
#       lo,up = N-i,N-i+1
#    else: 
#        print('Error in finding root')
#    return x[up] - (x[lo]-x[up])/(y[lo]-y[up])*y[up]
#
#def heap_comp_individual(rootFile = 'Sep19/' ,rootFileName = '',N=16000,T=30.0,TSep=None,NSep=None,h_min=0.0001,NuRe=1000,NuPe=None,FrSquared=2.828,U_s=0.020,h1init=1.0,h2init=0.7,c1init=1.0,c2init=0.7,apart=5.0,sharp=200,show = False,CFL=0.1, save = True, NoInteractionAnalysis = False,show_legend=False,title = False,forArticle = False):
#    details = buildFileName(N, T, h_min, NuRe, FrSquared, U_s,h1init=h1init,h2init=h2init,c1init=c1init,c2init=c2init,apart=apart,sharp=sharp,CFL=CFL)
#    d1 = unpack(rootFile + rootFileName + details + '/d1')[-1]
#    x,T,d2 = unpack(rootFile + rootFileName + details + '/d2')
#
#    d=d1[-1,:]+d2[-1,:]
#
#    #for i in range(len(T)):
#    #    plt.plot(x,d1[i,:]-d2[i,:],label = 't=%0.2f'%T[i])
#    #plt.legend()
#    #plt.show()
#    equal_conc_point = find_zero_center_out(x,d1[-1,:]-d2[-1,:]) 
#
#    r = equal_conc_point
#    coll_idx = np.argmin(np.abs(T-3))
#    D1 = d1[-1,:]-d1[coll_idx,:]
#    D2 = d2[-1,:]-d2[coll_idx,:]
#    plt.close()
#    plt.plot(x,D1)
#    plt.plot(x,D2)
#    plt.plot(x,d1[-1,:])
#    plt.plot(x,d2[-1,:])
#    plt.plot([r,r],[0,.1])
#    plt.xlim([-20,20])
#    plt.show()
#
#    left_max = np.max(d[x<0])
#    right_max = np.max(d[x>=0])
#    max_loc = x[np.argmax(d)]
#    return left_max, right_max, right_max/left_max, right_max-left_max, max_loc, equal_conc_point
#
##heap_comp_individual(rootFile = 'SedimentationInitialConditionTest_2025Apr25/',N=5000,CFL=0.1,U_s=0.010,T=40.,sharp=50)
#
#def heap_comp(H2 = np.linspace(0.7,0.97,10),C2 = np.linspace(0.7,0.97,10),rootFile = 'Sep17_loopy/' ,rootFileName = '',N=16000,T=30.0,TSep=None,NSep=None,h_min=0.0001,NuRe=1000,NuPe=None,FrSquared=2.828,U_s=0.010,h1init=1.0,h2init=0.7,c1init=1.0,c2init=0.7,apart=5.0,sharp=50,show = False,CFL=0.1, save = True, NoInteractionAnalysis = False,show_legend=False,title = False,forArticle = False):
#    H_mesh,C_mesh=np.meshgrid(H2,C2)
#    heap_mesh = H_mesh*0.
#    rel_mesh = H_mesh*0.
#    eq_conc_mesh = H_mesh*0.
#    for i,h2 in enumerate(H2):
#        for j,c2 in enumerate(C2):
#            
#            rel, loc, eq_conc = heap_comp_individual(rootFile = rootFile,rootFileName = rootFileName,N=N,T=T,TSep=TSep,NSep=NSep,h_min=h_min,NuRe=NuRe,NuPe=NuPe,FrSquared=FrSquared,U_s=U_s,h1init=h1init,h2init=h2,c1init=c1init,c2init=c2,apart=apart,sharp=sharp,show = show,CFL=CFL, save = save, NoInteractionAnalysis = NoInteractionAnalysis,show_legend=show_legend,title = show_legend,forArticle = forArticle)[-3:]
#            heap_mesh[j,i]=loc
#            rel_mesh[j,i]=rel
#            eq_conc_mesh[j,i]=eq_conc
#    plt.subplot(131)
#    fig=plt.pcolormesh(H_mesh,C_mesh,heap_mesh)
#    #fig=plt.pcolormesh(H_mesh,C_mesh,heap_mesh,shading = 'gouraud')
#    #plt.plot(H2,1/H2)
#    plt.colorbar(fig)
#    plt.xlabel('h2')
#    plt.ylabel('c2')
#
#    plt.subplot(132)
#    #fig=plt.pcolormesh(H_mesh,C_mesh,rel_mesh)
#    fig=plt.pcolormesh(H_mesh,C_mesh,rel_mesh,shading = 'gouraud')
#    plt.plot(H2,0.7*0.97/H2)
#    plt.colorbar(fig)
#    plt.xlabel('h2')
#    plt.ylabel('c2')
#
#    plt.subplot(133)
#    #fig=plt.pcolormesh(H_mesh,C_mesh,rel_mesh)
#    fig=plt.pcolormesh(H_mesh,C_mesh,eq_conc_mesh,shading = 'gouraud')
#    plt.plot(H2,0.7*0.97/H2)
#    plt.colorbar(fig)
#    plt.xlabel('h2')
#    plt.ylabel('c2')
#
#    plt.show()
##heap_comp(rootFile = 'SedimentationInitialConditionTest_2025Apr25/',N=5000,CFL=0.1,U_s=0.010,T=40.)
#
#def new_sediment(rootFile = 'Sep17_loopy/' ,rootFileName = '',N=16000,T=30.0,TSep=None,NSep=None,h_min=0.0001,NuRe=1000,NuPe=None,FrSquared=2.828,U_s=0.020,h1init=1.0,h2init=0.7,c1init=1.0,c2init=0.7,apart=5.0,sharp=200,show = False,CFL=0.1, save = True, NoInteractionAnalysis = False,show_legend=False,title = False,forArticle = False):
#    details = buildFileName(N, T, h_min, NuRe, FrSquared, U_s,h1init=h1init,h2init=h2init,c1init=c1init,c2init=c2init,apart=apart,sharp=sharp,CFL=CFL)
#    d1 = unpack(rootFile + rootFileName + details + '/d1')[-1]
#    x,T,d2 = unpack(rootFile + rootFileName + details + '/d2')
#
#    t_legend=[]
#    d1d2_old = d1[0,:]*0
#    if forArticle:
#        plt.rcParams.update({"text.usetex":True,"font.family": "sans-serif","font.sans-serif": ["Helvetica"],'font.size':12,'legend.fontsize':10,'lines.linewidth':0.5,'legend.fontsize':12,'xtick.labelsize':10,'ytick.labelsize':10})
#        plt.figure(figsize = [5*1.3,1.73*1.1])
#    for i,t in enumerate(T):
#        if i%4==0 and i>0:
#            dminus = d1[4,:]-d2[4,:]
#            polygon = plt.fill_between(x, d1d2_old, d1[i,:] + d2[i,:]-dminus, lw=1, color='none')
#            #ylim = plt.ylim()
#            verts = np.vstack([p.vertices for p in polygon.get_paths()])
#            gradient = plt.imshow(np.reshape((d1[i,:]-d1[4,:])/(d1[i,:]+d2[i,:]-dminus),(1,-1)), cmap='PRGn', aspect='auto',extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
#            gradient.set_clip_path(polygon.get_paths()[0], transform=plt.gca().transData)
#            #plt.ylim(ylim)
#            #plt.plot(x,d1[-1,:],color='k')
#            #plt.plot(x,d2[-1,:],color='k')
#            plt.plot(x,d1[i,:]+d2[i,:],color='k')
#            #        plt.plot(x,d1[i,:]+d2[i,:],label='t=%0.1f'%t)
#            #plt.legend()
#            d1d2_old = d1[i,:]+d2[i,:]
#            t_legend.append("t=%0.1f"%t)
#    t_legend.reverse()
#    plt.legend(t_legend,labelspacing=0)
#    cbar=plt.colorbar(gradient)
#    cbar.set_ticks(ticks=[0,1])
#    cbar.ax.set_yticklabels(['$100\%\ c_r$','$100\%\ c_{\ell}$'])
#    plt.xlim([-10,10])
#    if forArticle:
#        plt.subplots_adjust(bottom = 0.12, left = 0.07, top = 0.952, right = 1.03)
#    filename =  'solutions/plots/post_collition_h1_%0.2f_h2_%0.2f_c1_%0.2f_c2_%0.2f'%(h1init,h2init,c1init,c2init)
#    plt.savefig(rootFile + filename.replace('.','_') + '.png',dpi =800)
#    #plt.show()
#    plt.close()
#
##for c2 in np.linspace(0.7,0.97,10):
##    for h2 in np.linspace(0.7,0.97,10):
##for c2 in [0.7,0.97]:
##    for h2 in [0.7,0.97]:
##        new_sediment(forArticle=True,h1init=0.0,h2init=h2,c1init=0.0,c2init=c2,N=2000,sharp = 50)
#
#def sediment(rootFile = 'ResearchStatement/' ,rootFileName = '',N=16000,T=30.0,TSep=None,NSep=None,h_min=0.0001,NuRe=1000,NuPe=None,FrSquared=2.828,U_s=0.020,h1init=None,h2init=None,c1init=None,c2init=None,apart=None,sharp=200,show = False,CFL=0.1, save = True, NoInteractionAnalysis = False,show_legend=False,title = False,forArticle = False):
#    if not TSep:
#        TSep = T
#    if not NSep:
#        NSep = N
#    print('sediment:', ' c1=',c1init,' c2=',c2init,' h1=',h1init,'\n')
#    if NoInteractionAnalysis:
#        details = buildFileName(NSep, TSep, h_min, NuRe, FrSquared, U_s,h1init=h1init,h2init=0.0,c1init=c1init,c2init=0.0,apart=apart,sharp=sharp,CFL=CFL)
#        print(details)
#        xSep,_,h = unpack(rootFile + rootFileName + details + '/h')
#        H1_NO = h
#        temp = unpack(rootFile + rootFileName + details + '/phi1')[-1]
#        C1_NoInteraction = temp/h
#        details = buildFileName(NSep, TSep, h_min, NuRe, FrSquared, U_s,h1init=0.0,h2init=h2init,c1init=0.0,c2init=c2init,apart=apart,sharp=sharp,CFL=CFL)
#        print(details)
#        h = unpack(rootFile + rootFileName + details + '/h')[-1]
#        temp = unpack(rootFile + rootFileName + details + '/phi2')[-1]
#        C2_NoInteraction = temp/h
#        H2_NO = h
#        C_NoInteraction = C1_NoInteraction+C2_NoInteraction 
#        d_NoInteraction = C_NoInteraction[0,:]*0
#        D_NoInteraction = []
#    details = buildFileName(N, T, h_min, NuRe, FrSquared, U_s,h1init=h1init,h2init=h2init,c1init=c1init,c2init=c2init,apart=apart,sharp=sharp,CFL=CFL)
#    h = unpack(rootFile + rootFileName + details + '/h')[-1]
#    x,T,temp = unpack(rootFile + rootFileName + details + '/phi1')
#    C1 = temp/h
#    temp = unpack(rootFile + rootFileName + details + '/phi2')[-1]
#    C2 = temp/h
#    
#
#    d1 = C1[0,:]*0
#    d2 = C2[0,:]*0
#    dt = T[1]-T[0]
#    colorcounter = 0
#    if forArticle:
#        plt.rcParams.update({"text.usetex":True,"font.family": "sans-serif","font.sans-serif": ["Helvetica"],'font.size':12,'legend.fontsize':12,'lines.linewidth':0.5,'legend.fontsize':12,'xtick.labelsize':10,'ytick.labelsize':10})
#        plt.figure(figsize = [5*1.3,1.73*1.1])
#    else:
#        plt.figure(figsize = [15*1.3,4*1.3])
#    D1 = []
#    D2 = []
#    Tplot = []
#    print('running loop')
#    
#    if T[-1] == T[-2]:
#        T = T[:-1]
#    for i,t in enumerate(T):
#        d1 += dt*U_s*C1[i,:]
#        d2 += dt*U_s*C2[i,:]
#        if NoInteractionAnalysis and t<TSep:
#            d_NoInteraction += dt*U_s*C_NoInteraction[i,:]
#        if abs(t%5)<dt/2 and t>0:
#            D1.append(deepcopy(d1))
#            D2.append(deepcopy(d2))
#            if NoInteractionAnalysis:
#                D_NoInteraction.append(deepcopy(d_NoInteraction))
#            Tplot.append(t)
#    #plt.show()
#    print('loop complete')
#    if NoInteractionAnalysis:
#        D_NoInteraction.reverse()
#    D2.reverse()
#    D1.reverse()
#    Tplot.reverse()
#    print(Tplot)
#    opac = 1
#    d_opac = (opac-0.2)/len(Tplot)
#    print('start plotting')
#    for i in range(len(Tplot)):
#        d1 = D1[i]
#        d2 = D2[i]
#        t = Tplot[i]
#        d2layered = d1+d2
#        if show_legend:
#            plt.fill_between(x,d1+d2,facecolor = (0.7517*opac,0.7517*opac,0.7517*opac), edgecolor = (0,0,0,1),linewidth = 0.75,label = '%0.2f'%t)
#        plt.fill_between(x[d1>0],d1[d1>0],facecolor = (0.5803*opac,0.4040*opac,0.7412*opac), edgecolor = (0,0,0,1),linewidth = 2)
#        plt.fill_between(x[d2>0],d2layered[d2>0],y2 = d1[d2>0],facecolor = (1*opac,0.4980*opac,0.0549*opac), edgecolor = (0,0,0,1),linewidth = 1)
#        if NoInteractionAnalysis and t < TSep:
#             if forArticle:
#                 plt.plot(xSep,D_NoInteraction[i], color = (0,0.569,0.702),linewidth=1.5)
#             else:
#                 plt.plot(xSep,D_NoInteraction[i], color = (0,0.569,0.702),linewidth=3)
#        colorcounter+=1
#        opac -= d_opac
#        print(i)
#    print('plotting complete')
#    #plt.xlim((x[D1[0]>0][0],x[D2[0]>0][-1]))
#    plt.xlim((-10,10))
#    if title: plt.title('Sediment Deposition Patterns: c1=%0.1f, c2=%0.1f, h1=%0.1f, h2=%0.1f'%(c1init,c2init,h1init,h2init))
#    if forArticle:
#        plt.subplots_adjust(bottom = 0.12, left = 0.07, top = 0.99, right = .97)
#
#    if show_legend:
#        plt.legend()
#    if save:
#        print('save plot\n')
#        if NoInteractionAnalysis:
#            filename = 'layeredSedimentDepositsWithNoInteraction_cOne%0.1f_cTwo%0.1f_hOne%0.1f_hTwo%0.1f_T%0.1f'%(c1init,c2init,h1init,h2init,TSep)
#        else:
#            filename = 'layeredSedimentDeposits_cOne%0.1f_cTwo%0.1f_hOne%0.1f_hTwo%0.1f_T%0.1f'%(c1init,c2init,h1init,h2init,TSep)
#        plt.savefig(rootFile + 'solutions/plots/sediments/' + filename.replace('.','_') + '.pdf')
#        if not save: plt.close()
#    if show: plt.show()
#
#
#def runViscNonSymBurgers():
#    List = ['viscous_nonsymmetric_N1000_T4_nu_0_0','viscous_nonsymmetric_N1000_T4_nu_0_001','viscous_nonsymmetric_N1000_T4_nu_0_01','viscous_nonsymmetric_N1000_T4_nu_0_1']
#    Leg = ['nu = 0','nu = 0.001','nu = 0.01','nu = 0.1']
#    makeMP4withMultFiles(rootFile = 'Burgers/',filelist = List,Legend=Leg, SAVE = True)
#
#def avg_cell(x,u,t):
#    # u is a function, not a vector.
#    N = x.size + 1 # Number of nodes
#    u_avg = np.empty((N-1))
#    deltax=x[2]-x[1]
#    x = x - deltax/2
#    x = np.hstack((x,np.array([x[-1]+deltax]))) # Recreate nodes from cell midpoints
#    for j in np.arange(0,N-1):
#        # Make sure cell averaging is consistent with how the initial condition is averaged over every cell.
#        #u_avg[j] = np.dot(u(xg*deltax/2+(x[j]+x[j+1])/2,t),wg) 
#        u_avg[j] = (u(x[j],t) + u(x[j+1],t))/2
#    return u_avg
#
#def ErrorAnalysisWithTrueSolution():
#    """
#    Here is the error analysis for the test problem u_t + u_x = 0, -1<x<=1, t>0
#    The true solution is u(x,t) = exp(sin(pi(x-t)))
#    The initial condition is u(x,0)
#    """
#    rootFile = "Transport/"
#    
#    def true_soln(x,t):
#        return np.exp(np.sin(np.pi*(x-t)))
#
#    #N = range(60,220,20)
#    N = [40,60,80,100,120,140,160,180,200,300,400,500,600,700,800,900,1000]
#    for T in [0.5,1,1.5,2]:
#        E = []
#        for n in N:
#            x,t,u = unpack(rootFile + "exp_sin_pi_x_RK3_N" + str(n) + "_T2")
#            print(u.shape)
#            dx = x[1]-x[0]
#            index = np.argmin(np.abs(t-T))
#            t = t[index]
#            u = u[index,:]
#            u_ = avg_cell(x,true_soln,t) # Cell averaged "exact" solution.
#            E.append(dx*np.sum(np.abs(u-u_)))
#        P = np.polyfit(np.log(N),np.log(E),1)
#        plt.loglog(N,E,marker='s',label = 't = %0.2f, ROC: %0.2f'%(T,P[0]))
#    plt.legend()
#    plt.title('$u_t+u_x=0$, exact: $u(x,t) = \exp(\sin(\pi(x-t)))$')
#    plt.xlabel('N nodes')
#    plt.ylabel('$L^1$ error')
#    plt.show()
#
#def fronts(whichfile, timecutoff = 10000000, tolerance = 1e-1, rootFile = 'TwoCurrentShallowWaterSimulations/' ,rootFileName = 'BoxICs_',N=8000,T=30.0,h_min=0.001,NuRe=250,NuPe=250,FrSquared=2,U_s=0.01):
#    """
#    Use the cutoff idea to plot the fronts of the waves
#    """
#    details = buildFileName(N, T, h_min, NuRe, NuPe, FrSquared, U_s)
#    SWEFiles = rootFile + rootFileName + whichfile + details
#    x,t,u = unpack(SWEFiles)
#    LeftFront = []
#    RightFront = []
#    for i,c in zip(t,u): 
#        if i>timecutoff: break
#        print(i)
#        LeftFront.append(x[np.argwhere(c>tolerance)[0,0]]) # Find the first value where the variable of interest (c) is about the cutoff
#        RightFront.append(x[np.argwhere(c>tolerance)[-1,0]]) # Find the last value where the variable of interest (c) is about the cutoff
#    idx =  np.argwhere(t<timecutoff)[:,0] #If you cutoff the time values, you only want the relative information in t and u returned.
#    return x, t[idx], u[idx,:], np.array(LeftFront), np.array(RightFront)
#
#def plotFronts(var= 'c',timecutoff = 20, tolerance = 1e-4, rootFile = 'TwoCurrentShallowWaterSimulations/' ,rootFileName = 'BigBoxLittleBox_',N=8000,T=30.0,h_min=0.001,NuRe=250,NuPe=250,FrSquared=2,U_s=0.01,show = True,linestyle = 'solid',color = 'red'):
#    x,t,c1,c1Left, c1Right = fronts(var + '1',tolerance = tolerance, timecutoff=timecutoff,rootFileName = rootFileName, N = N, T = T, h_min = h_min, NuRe = NuRe, NuPe = NuPe, FrSquared = FrSquared, U_s = U_s)
#    x,t,c2,c2Left, c2Right = fronts(var + '2', tolerance = tolerance, timecutoff=timecutoff,rootFileName = rootFileName, N = N, T = T, h_min = h_min, NuRe = NuRe, NuPe = NuPe, FrSquared = FrSquared, U_s = U_s)
#    #plt.plot(t,c1Right)
#    #plt.plot(t,c2Left)
#    #plt.figure()
#    plt.plot(t,c1Right,color = color,linestyle =  'solid')
#    plt.plot(t,c1Left, color = color,linestyle =  'solid')
#    plt.plot(t,c2Right,color= color,linestyle = 'dashed')
#    plt.plot(t,c2Left,color = color,linestyle = 'dashed')
#    
#    if show: plt.show()
#
#def fronts_vs_Num(NuRe, NuPe, timecutoff = 20, tolerance = 1e-4, rootFile = 'TwoCurrentShallowWaterSimulations/' ,rootFileName = 'BigBoxLittleBox_',N=8000,T=30.0,h_min=0.001,FrSquared=2,U_s=0.01,show = True): 
#
#    color = ['red','blue','black','green','purple','orange']
#    for nR,nP,c in zip(NuRe,NuPe,color[:len(NuRe)]):
#        plotFronts(timecutoff = 30.0, rootFileName = rootFileName, U_s = U_s, tolerance = tolerance,rootFile = rootFile, T = T, h_min = h_min, FrSquared = FrSquared, NuRe = nR, NuPe = nP, color = c,show = False)
#
#    leg = plt.legend(NuRe)
# 
#    for i,j in enumerate(leg.legendHandles):
#        j.set_color(color[i])
#        j.set_linestyle('solid')
#    plt.show()
##fronts_vs_Num(NuRe=[62,125,250,500], NuPe=[62,125,250,500],rootFileName = 'BigBoxLittleBox_NuRePeTest_',U_s = 0.0)
#
#
#def trapz(x,f):
#    """
#    Trapezoidal rule to compute average value between the fronts
#
#    This does the average as well, not just the integration, see the division by (x[-1]-x[0]) in the return statement
#    """
#    dx = x[1]-x[0]
#    F = 0 
#    for i in range(len(x)-1): 
#        F += dx*(f[i]+f[i+1])/2
#    return F/(x[-1]-x[0]) 
#
#def boxify(whichfile, timecutoff = 1000000000, tolerance = 1e-4, rootFile = 'TwoCurrentShallowWaterSimulations/' ,rootFileName = 'BigBoxLittleBox_',N=8000,T=30.0,h_min=0.001,NuRe=250,NuPe=250,FrSquared=2,U_s=0.01):
#    x, time, C, cLeft, cRight = fronts(whichfile,tolerance = tolerance, timecutoff=timecutoff,rootFileName = rootFileName, N = N, T = T, h_min = h_min, NuRe = NuRe, NuPe = NuPe, FrSquared = FrSquared, U_s = U_s)
#    box = []
#    for c,cl,cr in zip(C, cLeft, cRight):
#        indeces = np.argwhere(np.all([list(x>=cl),list(x<=cr)],axis=0))[:,0]
#        box.append(trapz(x[indeces],c[indeces]))
#    return time, np.array(box), cLeft, cRight
#
#def boxifiedData(timecutoff = 20, tolerance = 1e-4, rootFile = 'TwoCurrentShallowWaterSimulations/' ,rootFileName = 'BigBoxLittleBox_',N=8000,T=30.0,h_min=0.001,NuRe=250,NuPe=250,FrSquared=2,U_s=0.00):
#    t, b1, cl1, cr1 = boxify('c1',tolerance = tolerance, timecutoff=timecutoff,rootFileName = rootFileName, N = N, T = T, h_min = h_min, NuRe = NuRe, NuPe = NuPe, FrSquared = FrSquared, U_s = U_s)
#    t, b2, cl2, cr2 = boxify('c2',tolerance = tolerance, timecutoff=timecutoff,rootFileName = rootFileName, N = N, T = T, h_min = h_min, NuRe = NuRe, NuPe = NuPe, FrSquared = FrSquared, U_s = U_s)
#    Filename = buildFileName(N, min(timecutoff,T), h_min, NuRe, NuPe, FrSquared, U_s)
#    np.savetxt(rootFile + 'solutions/TwoBoxifiedData1' + Filename + '_tolerance%0.6f'%tolerance + '.txt',np.array([t,b1,cl1,cr1,b2,cl2,cr2]).T)
#
#def tolSensAvgConc(whichfile,timecutoff = 30,rootFile = 'TwoCurrentShallowWaterSimulations/' ,rootFileName = 'BigBoxLittleBox_',N=8000,T=30.0,h_min=0.001,NuRe=250,NuPe=250,FrSquared=2,U_s=0.00):
#    TOL = [1e-3,1e-4,1e-5,1e-6]
#    #TOL = [0.0]
#    for tol in TOL:
#        t, b1, cl1, cr1 = boxify(whichfile,tolerance = tol, timecutoff=timecutoff,rootFileName = rootFileName, N = N, T = T, h_min = h_min, NuRe = NuRe, NuPe = NuPe, FrSquared = FrSquared, U_s = U_s)
#        plt.plot(t,b1,label=tol)
#    plt.legend()
#
#    
#def tolSens():
#    TOL = [1e-4,1e-4,1e-5,1e-5,1e-6,1e-6]
#    var = ['c','phi','c','phi','c','phi']
#    color = ['red','blue','black','green','purple','orange']
#    Lege = []
#    for t,v,c in zip(TOL,var,color[:len(TOL)]):
#        plotFronts(var = v, timecutoff = 30.0, rootFileName = 'SmoothICs_', U_s = 0.0, tolerance = t, color = c,show = False)
#        Lege.append(v + ', %0.6f'%t)
#
#
#    leg = plt.legend(Lege)
# 
#    for i,j in enumerate(leg.legendHandles):
#        j.set_color(color[i])
#        j.set_linestyle('solid')
#
#    plt.show()
#
#def oldVsNew(whichfile, N=500,T=2.0, h_min = 0.0001, NuRe = 2000, FrSquared = 2, U_s = 0.00):
#    """
#    Run the old versus new code and plot the final time to check new code.
#    """
#    details = buildFileName(N=N, T=T, h_min=h_min, NuRe=NuRe, FrSquared=FrSquared, U_s=U_s)
#    xnew,tnew,unew = unpack('TwoCurrentShallowWaterSimulations/Sep12Test_' + whichfile + details)
#    details = buildFileName(N=N, T=T, h_min=h_min, NuRe=NuRe, FrSquared=FrSquared, U_s=U_s)
#    xold,told,uold = unpack('TwoCurrentShallowWaterSimulations/Sep12TestOLD_' + whichfile + details)
#    plt.plot(xnew,unew[-1,:],label = 'new')
#    plt.plot(xold,uold[-1,:],label = 'old',linestyle = 'dashed')
#    plt.title(whichfile)
#    plt.legend()
#
#def oldVsNewAll():
#    plt.subplot(221)
#    oldVsNew('u')
#    plt.subplot(222)
#    oldVsNew('h')
#    plt.subplot(223)
#    oldVsNew('c1')
#    plt.subplot(224)
#    oldVsNew('c2')
#    plt.show()
#
#def plot_time(T,x,t,u,label):
#    index = np.argmin(np.abs(t-T))
#    plt.plot(x,u[index,:],label = label)
#    
#def plot_timenearfront(T,x,t,u,label):
#    index = np.argmin(np.abs(t-T))
#    U = u[index,:]
#    FRONT = x[np.argwhere(U>U[-2])[-1][0]]
#    idx = np.argwhere(np.all([x>FRONT-1,x<FRONT+1],axis=0))
#    plt.plot(x[idx],U[idx],label = label)
#
#def nearfront(T,x,t,u):
#    index = np.argmin(np.abs(t-T))
#    U = u[index,:]
#    FRONT = x[np.argwhere(U>U[-2])[-1][0]]
#    idx = np.argwhere(np.all([x>FRONT-1,x<FRONT+1],axis=0))
#    return x[idx],U[idx]
#
#def max_versus_h_min(whichfile = 'h', n = 5,U_s = 0.0,NuRe= 1000):
#    h_min=[0.00006,0.00007,0.00008,0.00009,0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001]
#    maxh = []
#    for h in h_min:
#        details = buildFileName(N=1000, T=20.0, h_min=h, NuRe=NuRe, FrSquared=2, U_s=U_s)
#        x,t,u = unpack('h_min_test/Sep29_' + whichfile + details)
#        Xx,U = nearfront(10,x,t,u)
#        maxh.append(np.max(U))
#    
#    plt.plot(h_min,maxh,marker = 'd')
#    #plt.savefig('h_min_test/plots/h_versus_maxloglog.pdf')
#    p = np.polyfit(maxh,h_min,2)
#    #p3 = np.polyfit(h_min,maxh,3)
#    #p4 = np.polyfit(h_min,maxh,4)
#    x = np.linspace(1.03,1.25)
#    plt.plot(p[0]*x**2 + p[1]*x + p[2],x,linestyle = 'dashed')
#    plt.show()
#
#def test_h_min(h_min,whichfile = 'h', n = 5,U_s = 0.0,NuRe= 1000):
#    """
#    Plot the solution for n evenly spaced points in time
#    """
#    plt.figure(figsize = [12,10])
#    for h in h_min:
#        details = buildFileName(N=1000, T=20.0, h_min=h, NuRe=NuRe, FrSquared=2, U_s=U_s)
#        x,t,u = unpack('h_min_test/Sep29_' + whichfile + details)
#        plt.subplot(211)
#        plot_time(10,x,t,u,label = 'hm = %0.5f'%h)
#        plt.title(variable_dict[whichfile[0]] + ', time = %0.2f'%10)
#        plt.legend()
#        plt.subplot(212)
#        plot_time(20,x,t,u,label = 'hm = %0.5f'%h)
#        plt.title(variable_dict[whichfile[0]] + ', time = %0.2f'%20)
#        plt.legend()
#
#def test_h_min_nearfront(h_min,whichfile = 'h', n = 5,U_s = 0.0,NuRe= 1000):
#    """
#    Plot the solution for n evenly spaced points in time
#    """
#    plt.figure(figsize = [12,10])
#    for h in h_min:
#        details = buildFileName(N=1000, T=20.0, h_min=h, NuRe=NuRe, FrSquared=2, U_s=U_s)
#        x,t,u = unpack('h_min_test/Sep29_' + whichfile + details)
#        plt.subplot(211)
#        plot_timenearfront(10,x,t,u,label = 'hm = %0.5f'%h)
#        plt.title(variable_dict[whichfile[0]] + ', time = %0.2f'%10)
#        plt.legend()
#        plt.subplot(212)
#        plot_timenearfront(20,x,t,u,label = 'hm = %0.5f'%h)
#        plt.title(variable_dict[whichfile[0]] + ', time = %0.2f'%20)
#        plt.legend()
#
#def example_timeplots(rootFile = 'TwoCurrentShallowWaterSimulations/InitialConditionTest/' ,rootFileName = '',N=5000,T = 15.0, whichfile = 'c1', n = 5,U_s = 0.00,h_min=0.0001,NuRe= 1000,FrSquared = 2,c1init = None, c2init = None, h1init = None, h2init = None, Nt =None, Ts = 0,linestyle = 'solid',show_legend = True,apart = 5, CFL=0.1,sharp = 50):
#    """
#    Plot the solution for n evenly spaced points in time
#    """
#    details = buildFileName(N=N, T=T, h_min=h_min, NuRe=NuRe, FrSquared=FrSquared, U_s=U_s,h1init=h1init, h2init=h2init, c1init=c1init, c2init=c2init, apart=apart, CFL=CFL, sharp=sharp)
#    if whichfile == 'u':
#        x,t,temp = unpack(rootFile + rootFileName + details + '/q')
#        h = unpack(rootFile + rootFileName + details + '/h')[-1]
#        u = temp/h
#    elif whichfile == 'c1':
#        x,t,temp = unpack(rootFile + rootFileName + details + '/phi1')
#        h = unpack(rootFile + rootFileName + details + '/h')[-1]
#        u = temp/h
#    elif whichfile == 'c2':
#        x,t,temp = unpack(rootFile + rootFileName + details + '/phi2')
#        h = unpack(rootFile + rootFileName + details + '/h')[-1]
#        u = temp/h
#    else:
#        x,t,u = unpack(rootFile + rootFileName + details + '/' + whichfile)
#    print(rootFile + rootFileName + whichfile + details)
#  
#    if not Nt:
#        Nt = len(t)
#    else:
#        Nt = sum(t<=Nt)
#    Ns = np.argmin(np.abs(t-Ts))
#    j = Ns
#    counter = 0
#    while j<=Nt:
#        plt.plot(x,u[j,:], label = 't = %0.1f'%t[j],linestyle = linestyle,color = tabcolors[counter])
#        if whichfile[0] == 'c':
#            plt.ylim((0,1))
#        plt.title(variable_dict[whichfile])
#        #plt.xlabel('x')
#        j+=int((Nt-Ns)/n)
#        counter+=1
#    if show_legend:
#        plt.legend()
#    #plt.show()
#
#def analyze_solutions(rootFile='TwoCurrentShallowWaterSimulations/InitialConditionTest/', rootFileName='Separated_', N=5000, T=15.0, U_s=0.02, h_min=0.0001, NuRe=1000, FrSquared=2, c1init=None, c2init=None, h1init=None, h2init=None, linestyle = 'solid'):
#    """
#    Plot the solution for n evenly spaced points in time
#    """
#    details = buildFileName(N=N, T=T, h_min=h_min, NuRe=NuRe, FrSquared=FrSquared, U_s=U_s,h1init=h1init, h2init=h2init, c1init=c1init, c2init=c2init)
#    x,t,h = unpack(rootFile + rootFileName + 'h' + details)
#    q = unpack(rootFile + rootFileName + 'q' + details)[-1]
#    #phi1 = unpack(rootFile + rootFileName + 'phi1' + details)[-1]
#    #phi2 = unpack(rootFile + rootFileName + 'phi2' + details)[-1]
#    u = q/h
#    j=0
#    while np.max(h[j,:]) >= np.max(h[j+1,:]):
#        collision_time = t[j]
#        collision_space_index = np.argmax(h[j+1,:]) 
#        collision_loc = x[collision_space_index]
#        collision_j = j
#        j+=1
#    momentum_spike = 0
#    check_time = True
#    while j <len(t):
#        #momentum_spike = max(momentum_spike,np.max(h[j,:]))
#        if np.max(h[j,:])>momentum_spike:
#            momentum_spike = np.max(h[j,:])
#            collision_space_index = np.argmax(h[j,:]) 
#            collision_loc = x[collision_space_index]
#        if t[j] > collision_time + 4 and check_time:
#            cutoff_value = (3*h[j,collision_space_index] + h[j,np.argwhere(x>x[np.argwhere(h[j,:]>3*h_min)[-1]]-1)[0]])/4
#            right_reflecting_wave_loc = x[np.argwhere(h[j,:]>cutoff_value)[-1]]
#            check_time = False
#        j+=1
#    return collision_time, collision_loc, momentum_spike, right_reflecting_wave_loc[0]
#
#def RH_analysis(rootFile='TwoCurrentShallowWaterSimulations/InitialConditionTest/', rootFileName='Separated_', N=5000, T=15.0, U_s=0.00, h_min=0.0001, NuRe=1000, FrSquared=2, c1init=None, c2init=None, h1init=None, h2init=None, linestyle = 'solid',show = False, save = False):
#    """
#    Plot the solution for n evenly spaced points in time
#    """
#    print('RH:', ' c1=',c1init,' c2=',c2init,' h1=',h1init,'\n')
#    def RH_h(hp,hm,up,um):
#        return (hp*up - hm*um)/(hp-hm)
#    def RH_u(hp,hm,up,um):
#        return (hp*up*up + hp*hp/(2*FrSquared) - hm*um*um - hm*hm/(2*FrSquared))/(hp*up - hm*um)
#    details = buildFileName(N=N, T=T, h_min=h_min, NuRe=NuRe, FrSquared=FrSquared, U_s=U_s,h1init=h1init, h2init=h2init, c1init=c1init, c2init=c2init)
#    x,t,h = unpack(rootFile + rootFileName + 'h' + details)
#    q = unpack(rootFile + rootFileName + 'q' + details)[-1]
#    u = q/h
#    j=0
#    while np.max(h[j,:]) >= np.max(h[j+1,:]):
#        collision_time = t[j]
#        collision_space_index = np.argmax(h[j+1,:]) 
#        collision_loc = x[collision_space_index]
#        collision_j = j
#        j+=1
#    momentum_spike = 0
#    R = []
#    S_h = []
#    S_u = []
#    t_new = []
#    while j <len(t):
#        if np.max(h[j,:])>momentum_spike:
#            momentum_spike = np.max(h[j,:])
#            collision_space_index = np.argmax(h[j,:]) 
#            collision_loc = x[collision_space_index]
#            R=[]
#            S_h = []
#            S_u = []
#            t_new = []
#        if t[j] > collision_time+1.5:
#            cutoff_value = (3*h[j,collision_space_index] + h[j,np.argwhere(x>x[np.argwhere(h[j,:]>3*h_min)[-1]]-1)[0]])/4
#            reflect_wave_index =np.argwhere(h[j,:]>cutoff_value)[-1][0] 
#            R.append(x[reflect_wave_index])
#            hp = np.polyfit(x[reflect_wave_index+10:reflect_wave_index+30],u[j,reflect_wave_index+10:reflect_wave_index+30],0)[0]
#            hm = np.polyfit(x[reflect_wave_index-30:reflect_wave_index-10],u[j,reflect_wave_index-30:reflect_wave_index-10],0)[0]
#            hp = h[j,reflect_wave_index+10]
#            hm = h[j,reflect_wave_index-10]
#            polyright = np.polyfit(x[reflect_wave_index+10:reflect_wave_index+30],u[j,reflect_wave_index+10:reflect_wave_index+30],1)
#            polyleft = np.polyfit(x[reflect_wave_index-30:reflect_wave_index-10],u[j,reflect_wave_index-30:reflect_wave_index-10],1)
#            up = polyright[0]*x[reflect_wave_index] + polyright[1]
#            um = polyleft[0]*x[reflect_wave_index] + polyleft[1]
#            S_h.append(RH_h(hp,hm,up,um))
#            S_u.append(RH_u(hp,hm,up,um))
#            t_new.append(t[j])
#        j+=1
#    h_shock = R[0]
#    H_shock = []
#    u_shock = R[0]
#    U_shock = []
#    dt = t_new[1] - t_new[0]
#    for ssh,ssu in zip(S_h,S_u):
#        H_shock.append(h_shock)
#        h_shock += dt*ssh
#        U_shock.append(u_shock)
#        u_shock += dt*ssu
#    plt.figure(figsize=[6,4.5])
#    plt.plot(t_new,R,label = 'front tracking',color = 'tab:blue')
#    plt.plot(t_new,H_shock, label = 'S_h(t)',linestyle = 'dashed',color = 'tab:red')
#    plt.plot(t_new,U_shock, label = 'S_u(t)',linestyle = '-.',color = 'tab:green')
#    plt.title('c1=%0.1f, c2=%0.1f, h1=%0.1f, h2=%0.1f'%(c1init,c2init,h1init,h2init))
#    plt.legend()
#    if save:
#        filename = 'reflectingWaveRH_cOne%0.1f_cTwo%0.1f_hOne%0.1f_hTwo%0.1f'%(c1init,c2init,h1init,h2init)
#        plt.savefig(rootFile + 'solutions/plots/RH/' + filename.replace('.','_') + '.pdf')
#        if not show: plt.close()
#    if show: plt.show()
#
#
#def heatmap_analysis(rootFile='TwoCurrentShallowWaterSimulations/InitialConditionTest/', rootFileName='Separated_', N=5000, T=15.0, U_s=0.0, h_min=0.0001, NuRe=1000, FrSquared=2):
#    conc1 = [0.7,0.8,0.9,1.0]
#    conc2 = [0.7,0.8,0.9,1.0]
#    height1 = [0.7,0.8,0.9,1.0]
#    h2 = 1.0
#    Collision_Time = np.zeros((len(conc1),len(conc2)))
#    Collision_Loc = np.zeros((len(conc1),len(conc2)))
#    Momentum_Spike = np.zeros((len(conc1),len(conc2)))
#    Reflect_Wave = np.zeros((len(conc1),len(conc2)))
#    for h1 in height1:
#        for i,c1 in enumerate(conc1):
#            for j,c2 in enumerate(conc2):
#                print(h1,c1,c2)
#                Collision_Time[i,j], Collision_Loc[i,j], Momentum_Spike[i,j], Reflect_Wave[i,j] = analyze_solutions(rootFile=rootFile,rootFileName = rootFileName, N=N, T=T, U_s=U_s,h_min=h_min,NuRe=NuRe,FrSquared=2,c1init=c1,c2init=c2,h1init=h1,h2init=h2)
#        plt.figure()
#        ax=sns.heatmap(Collision_Time,xticklabels = conc2,yticklabels = conc1)
#        ax.set(xlabel='c2', ylabel='c1')
#        ax.set_title('Collision Time, h1/h2 = %0.1f'%h1)
#        filename = 'solutions/plots/Heatmap/CollisionTimeHeatmap_h1%0.1f'%h1
#        plt.savefig(rootFile + filename.replace('.','_') + '.pdf')
#        plt.close()
#        
#        plt.figure()
#        ax=sns.heatmap(Collision_Loc,xticklabels = conc2,yticklabels = conc1)
#        ax.set(xlabel='c2', ylabel='c1')
#        ax.set_title('Collision Loc, h1/h2 = %0.1f'%h1)
#        filename = 'solutions/plots/Heatmap/CollisionLocHeatmap_h1%0.1f'%h1
#        plt.savefig(rootFile + filename.replace('.','_') + '.pdf')
#        plt.close()
#        
#        plt.figure()
#        ax=sns.heatmap(Momentum_Spike,xticklabels = conc2,yticklabels = conc1)
#        ax.set(xlabel='c2', ylabel='c1')
#        ax.set_title('Momentum Spike, h1/h2 = %0.1f'%h1)
#        filename = 'solutions/plots/Heatmap/MomentumSpikeHeatmap_h1%0.1f'%h1
#        plt.savefig(rootFile + filename.replace('.','_') + '.pdf')
#        plt.close()
#        
#        plt.figure()
#        ax=sns.heatmap(Reflect_Wave,xticklabels = conc2,yticklabels = conc1)
#        ax.set(xlabel='c2', ylabel='c1')
#        ax.set_title('Reflect Wave, h1/h2 = %0.1f'%h1)
#        filename = 'solutions/plots/Heatmap/ReflectWaveHeatmap_h1%0.1f'%h1
#        plt.savefig(rootFile + filename.replace('.','_') + '.pdf')
#        plt.close()
#
#def plotsplotsplots(c1init=1.0,c2init=0.9,h1init=1.0,h2init=0.7, rootFile='Oct24/',save = False, show = False, Nt=12,Ts=0, show_legend = True,N=8000,apart=5,CFL=0.1,T=15.0,NuRe=1000,FrSquared=2.828,U_s=0.02,h_min=0.0001,sharp=50):
#    plt.figure(figsize=[12,9])
#    plt.subplot(411)
#    example_timeplots(rootFile = rootFile,whichfile = 'h',n=3,Nt = Nt,c1init = c1init, c2init = c2init, h1init = h1init, h2init = h2init, linestyle = 'solid',show_legend = show_legend,N=8000,apart=5,CFL=0.1,T=15.0,NuRe=1000,FrSquared=2.828,U_s=0.02,h_min=0.0001,sharp=50)
#    plt.xlim([-15,15])
#    plt.subplot(412)
#    example_timeplots(rootFile = rootFile,whichfile = 'u',n=3,Nt = Nt,c1init = c1init, c2init = c2init , h1init = h1init, h2init = h2init,linestyle = 'solid',show_legend = show_legend,N=8000,apart=5,CFL=0.1,T=15.0,NuRe=1000,FrSquared=2.828,U_s=0.02,h_min=0.0001,sharp=50)
#    plt.xlim([-15,15])
#    plt.subplot(413)
#    example_timeplots(rootFile = rootFile,whichfile = 'c1',n=3,Nt = Nt,c1init = c1init, c2init = c2init, h1init = h1init, h2init = h2init,linestyle = 'solid',show_legend = show_legend,N=8000,apart=5,CFL=0.1,T=15.0,NuRe=1000,FrSquared=2.828,U_s=0.02,h_min=0.0001,sharp=50)
#    plt.xlim([-15,15])
#    plt.subplot(414)
#    example_timeplots(rootFile = rootFile,whichfile = 'c2',n=3,Nt = Nt,c1init = c1init, c2init = c2init , h1init = h1init, h2init = h2init,linestyle = 'solid',show_legend = show_legend,N=8000,apart=5,CFL=0.1,T=15.0,NuRe=1000,FrSquared=2.828,U_s=0.02,h_min=0.0001,sharp=50)
#    plt.xlim([-15,15])
#    plt.xlabel('x')
#    plt.tight_layout()
#    if save:
#        filename = 'front_schematic_all_separate_cOne%0.1f_cTwo%0.1f_hOne%0.1f_hTwo%0.1f'%(c1init,c2init,h1init,h2init)
#        plt.savefig(rootFile + 'solutions/plots/' + filename.replace('.','_') + '.pdf')
#        if not save: plt.close()
#    if show: plt.show()
#    plt.figure(figsize=[12,6])
#    example_timeplots(rootFile = rootFile,whichfile = 'h',n=3,Nt = Nt,c1init = c1init, c2init = c2init, h1init = h1init, h2init = h2init, linestyle = 'solid',show_legend = show_legend,N=8000,apart=5,CFL=0.1,T=15.0,NuRe=1000,FrSquared=2.828,U_s=0.02,h_min=0.0001,sharp=50)
#
#    plt.xlim([-15,15])
#    if save:
#        filename = 'front_schematic_h_cOne%0.1f_cTwo%0.1f_hOne%0.1f_hTwo%0.1f'%(c1init,c2init,h1init,h2init)
#        plt.savefig(rootFile + 'solutions/plots/' + filename.replace('.','_') + '.pdf')
#        if not save: plt.close()
#    if show: plt.show()
#
#    plt.figure(figsize=[12,6])
#    example_timeplots(rootFile = rootFile,whichfile = 'h',n=4,Nt = Nt,Ts=4,c1init = c1init, c2init = c2init, h1init = h1init, h2init = h2init, linestyle = 'solid',show_legend = show_legend,N=8000,apart=5,CFL=0.1,T=15.0,NuRe=1000,FrSquared=2.828,U_s=0.02,h_min=0.0001,sharp=50)
#
#    plt.xlim([0,5])
#    if save:
#        filename = 'ZOOMEDfront_schematic_h_cOne%0.1f_cTwo%0.1f_hOne%0.1f_hTwo%0.1f'%(c1init,c2init,h1init,h2init)
#        plt.savefig(rootFile + 'solutions/plots/' + filename.replace('.','_') + '.pdf')
#        if not save: plt.close()
#    if show: plt.show()
#
#
#def initialCond(rootFile='Oct24/', rootFileName='', varList=['h','u','c1','c2'],Legend= None, SAVE = False, ymax = 1,ymin=0.1,show_legend = True,N=8000,T=15.0,h_min=0.0001,NuRe=1000,NuPe=None,FrSquared=2.828,U_s=0.020,h1init=1.0,h2init=0.7,c1init=1.0,c2init=0.9,apart=5,sharp = 50, CFL =0.1):
#    if not Legend:
#        leg = []
#    if not Legend:
#        Legend = leg
#    X = []
#    U = []
#    details = buildFileName(N=N, T=T, h_min=h_min, NuRe=NuRe, FrSquared=FrSquared, U_s=U_s,h1init=h1init, h2init=h2init, c1init=c1init, c2init=c2init,apart=apart,CFL=CFL,sharp=sharp)
#    for whichfile in varList:
#        if whichfile == 'u':
#            x,t,temp = unpack(rootFile + rootFileName + details + '/q')
#            h = unpack(rootFile + rootFileName + details + '/h')[-1]
#            u = temp/h
#        elif whichfile == 'c1':
#            x,t,temp = unpack(rootFile + rootFileName + details + '/phi1')
#            h = unpack(rootFile + rootFileName + details + '/h')[-1]
#            u = temp/h
#        elif whichfile == 'c2':
#            x,t,temp = unpack(rootFile + rootFileName + details + '/phi2')
#            h = unpack(rootFile + rootFileName + details + '/h')[-1]
#            u = temp/h
#        else:
#            x,t,u = unpack(rootFile + rootFileName + details + '/' + whichfile)
#        X.append(x)
#        U.append(u)
#    fig = plt.figure(figsize=(12,8))
#
#    colorlist = ['tab:blue','tab:red','tab:green','black','orange','darkviolet','slategray']
#    stylelist = ['solid','-.','dashed','dotted']
#    for ii in range(1):
#        subplotcounter = 1
#        for jj in range(len(varList)):
#            subplotcounter += 1
#            x = X[jj]
#            u = U[jj]
#            plt.subplot(2,1,int(subplotcounter/2))
#            plot_ = plt.plot(x,u[ii,:],color=colorlist[jj],linestyle = stylelist[0])
#    plt.ylim([0,1.05])
#    plt.subplot(211)
#    plt.ylim([0,1.05])
#    #plt.xlim([-17,17])
#    
#    plt.savefig(rootFile + 'solutions/plots/' + 'IC' + details.replace('.','_') + '.pdf')
#initialCond()
#
#def makeDFDplots(which = 'all', save = True, show = False):
#    if which == 'all' or which == 'sediment':
#        print('sediment')
#        sediment(c1init=0.7, c2init=1.0, h1init=1.0, h2init=1.0, save = save, show = show, NoInteractionAnalysis = False)
#        sediment(c1init=0.7, c2init=1.0, h1init=1.0, h2init=0.7, save = save, show = show, NoInteractionAnalysis = False)
#        sediment(c1init=1.0, c2init=1.0, h1init=1.0, h2init=1.0, save = save, show = show, NoInteractionAnalysis = False)
#        sediment(c1init=0.7, c2init=1.0, h1init=0.7, h2init=1.0, save = save, show = show, NoInteractionAnalysis = False)
#    if which == 'all' or which == 'sedimentWithNoInteraction':
#        print('sediment with no interaction')
#        sediment(c1init=0.7, c2init=1.0, h1init=1.0, h2init=1.0, save = save, show = show, NoInteractionAnalysis = True)
#        sediment(c1init=0.7, c2init=1.0, h1init=1.0, h2init=0.7, save = save, show = show, NoInteractionAnalysis = True)
#        sediment(c1init=1.0, c2init=1.0, h1init=1.0, h2init=1.0, save = save, show = show, NoInteractionAnalysis = True)
#    if which == 'all' or which == 'profile':
#        print('profile')
#        plotsplotsplots(0.8,0.9,1.0,0.7,show = False,save=True,Nt=12,show_legend = False)
#    if which == 'all' or which == 'RH':
#        print('Rankine-Hugoniot')
#        RH_analysis(c1init = 1.0,c2init = 0.7, h1init = 1.0, h2init = 0.7,save = True)
#        RH_analysis(c1init = 0.7,c2init = 0.7, h1init = 1.0, h2init = 0.7,save = True)
#        RH_analysis(c1init = 0.7,c2init = 0.7, h1init = 1.0, h2init = 1.0,save = True)
#    if which == 'all' or which == 'IC':
#        print('Initial Conditions')
#        initialCond(h1init = 1.0,h2init=0.7,c1init = 0.8, c2init = 0.9)
#    if which == 'all' or which == 'collisionVideo':
#        print('Videos')
#        runTwoCurrentSWE(c1init=0.8, c2init=0.9,h1init=1.0, h2init = 0.7)
#        runTwoCurrentSWE(rootFileName = 'Separated', c1init=0.8, c2init=0.9,h1init=1.0, h2init = 0.7,U_s=0.02)
#
#def spacetime(whichfile,leftBound = None,rightBound = None, maxT = None, minT=None, rootFile = 'TwoCurrentShallowWaterSimulations/InitialConditionTest/' ,rootFileName = 'Separated',N=4000,T=15.0,h_min=0.0001,NuRe=1000,NuPe=None,FrSquared=2,U_s=0.020,h1init=1.0,h2init=1.0,c1init=1.0,c2init=1.0,apart=5,CFL=0.1,sharp=50,show = False, save = True,contours=False,close=True,ZOOMED=False):
#    details = buildFileName(N=N, T=T, h_min=h_min, NuRe=NuRe, FrSquared=FrSquared, U_s=U_s,h1init=h1init, h2init=h2init, c1init=c1init, c2init=c2init,apart=apart,CFL=CFL,sharp=sharp)
#    if whichfile == 'u':
#        x,t,temp = unpack(rootFile + rootFileName + details + '/q')
#        h = unpack(rootFile + rootFileName + details + '/h')[-1]
#        func = temp/h
#    elif whichfile == 'c1':
#        x,t,temp = unpack(rootFile + rootFileName + details + '/phi1')
#        h = unpack(rootFile + rootFileName + details + '/h')[-1]
#        func = temp/h
#    elif whichfile == 'c2':
#        x,t,temp = unpack(rootFile + rootFileName + details + '/phi2')
#        h = unpack(rootFile + rootFileName + details + '/h')[-1]
#        func = temp/h
#    else:
#        x,t,func = unpack(rootFile + rootFileName + details + '/' + whichfile)
#    if not leftBound:  leftBound  = x[0]
#    if not rightBound: rightBound = x[-1]
#    if not maxT:       maxT  = t[-1]
#    if not minT:       minT  = t[0]
#    xMaxIdx = np.where(x<=rightBound)[-1][-1]
#    xMinIdx = np.where(x>= leftBound)[-1][0]
#    tMaxIdx = np.where(t<maxT)[-1][-1]
#    tMinIdx = np.where(t>minT)[-1][0]
#    xmesh,tmesh = np.meshgrid(x[xMinIdx:xMaxIdx],t[tMinIdx:tMaxIdx])
#    
#    if ZOOMED:
#        plt.figure(figsize=[3,6])
#    else:
#        plt.figure(figsize=[8,6])
#    fig = plt.pcolormesh(xmesh,tmesh,func[tMinIdx:tMaxIdx,xMinIdx:xMaxIdx],shading = 'gouraud',linewidth =0)
#    plt.xlabel('x')
#    plt.ylabel('time')
#    if ZOOMED:
#        plt.subplots_adjust(left = 0.3, right =0.98, top = 0.97, bottom =0.13)
#    else:
#        plt.subplots_adjust(left = 0.1, right =0.99, top = 0.97, bottom =0.13)
#        plt.colorbar(fig)
#    #if whichfile == 'u' or whichfile == 'q':
#    #    plt.set_cmap(cm['PuOr'])
#    #else:
#    #    plt.set_cmap(cm['viridis'])
#    if contours:
#        cp = plt.contour(xmesh,tmesh,func[:tMaxIdx,xMinIdx:xMaxIdx],20,colors = 'k')
#    if save:
#        if ZOOMED:
#            FileName = rootFile + 'solutions/plots/' + 'ZOOMEDspacetime%s_maxT%0.1f_domainwidth%0.1f_'%(whichfile,maxT,rightBound - leftBound) + details
#        else:
#            FileName = rootFile + 'solutions/plots/' + 'spacetime%s_maxT%0.1f_domainwidth%0.1f_'%(whichfile,maxT,rightBound - leftBound) + details
#        plt.savefig(FileName.replace('.','_') + '.pdf')
#        plt.savefig(FileName.replace('.','_') + '.png')
#    if show:
#        plt.show()
#    plt.close()
#
#def MaxVelvsTime(rootFile,rootFileName,fileDetails,FinalDistance):
#    X,T,temp = unpack(rootFile + rootFileName + 'q' + fileDetails)
#    h = unpack(rootFile + rootFileName + 'h' + fileDetails)[-1]
#    u = temp/h
#    uMax = []
#
#    h_min = h[0,-1] if h[0,0]==h[0,-1] else 2*float(fileDetails[fileDetails.find('hmin')+4:fileDetails.find('hmin')+11])
#    
#    for ii in range(len(T)):
#        if X[h[ii,:]>2*h_min][-1]>FinalDistance:
#            break
#        uMax.append(np.max(u[ii,:]))
#    print(X[h[ii,:]>2*h_min][-1],T[ii])
#    return T[:ii],np.array(uMax)
#
#
#def MaxVelvsDistance(rootFile,rootFileName,fileDetails,FinalDistance):
#    X,T,temp = unpack(rootFile + rootFileName + 'q' + fileDetails)
#    h = unpack(rootFile + rootFileName + 'h' + fileDetails)[-1]
#    u = temp/h
#    uMax = []
#
#    h_min = h[0,-1] if h[0,0]==h[0,-1] else 2*float(fileDetails[fileDetails.find('hmin')+4:fileDetails.find('hmin')+11])
#    
#    rightFront = []
#    for ii in range(len(T)):
#        if X[h[ii,:]>2*h_min][-1]>FinalDistance:
#            break
#            
#        rightFront.append(X[h[ii,:]>2*h_min][-1])
#        if T[ii]>0.5:
#            u_at_front = u[ii,list(np.all([X<rightFront[-1]+1,X>rightFront[-1]-1],axis=0))]
#            uMax.append(np.max(u_at_front))
#        else:
#            uMax.append(np.max(u[ii,:]))
#    print(X[h[ii,:]>2*h_min][-1],T[ii])
#    return np.array(rightFront),np.array(uMax)
#
#def FrontHeightvsDistance(rootFile,rootFileName,fileDetails,FinalDistance):
#    X,T,temp = unpack(rootFile + rootFileName + 'q' + fileDetails)
#    h = unpack(rootFile + rootFileName + 'h' + fileDetails)[-1]
#    u = temp/h
#    uMax = []
#
#    h_min = h[0,-1] if h[0,0]==h[0,-1] else 2*float(fileDetails[fileDetails.find('hmin')+4:fileDetails.find('hmin')+11])
#    
#    rightFront = []
#    for ii in range(len(T)):
#        if X[h[ii,:]>2*h_min][-1]>FinalDistance:
#            break
#            
#        rightFront.append(X[h[ii,:]>2*h_min][-1])
#        if T[ii]>0.5:
#            u_at_front = u[ii,list(np.all([X<rightFront[-1]+1,X>rightFront[-1]-1],axis=0))]
#            uMax.append(np.max(u_at_front))
#        else:
#            uMax.append(np.max(u[ii,:]))
#    print(X[h[ii,:]>2*h_min][-1],T[ii])
#    return np.array(rightFront),np.array(uMax)
#
#def FrontHeightvsTime(rootFile,rootFileName,fileDetails,FinalDistance):
#    X,T,temp = unpack(rootFile + rootFileName + 'q' + fileDetails)
#    h = unpack(rootFile + rootFileName + 'h' + fileDetails)[-1]
#    u = temp/h
#    FrontHeight = []
#
#    h_min = h[0,-1] if h[0,0]==h[0,-1] else 2*float(fileDetails[fileDetails.find('hmin')+4:fileDetails.find('hmin')+11])
#    
#    for ii in range(len(T)):
#        if X[h[ii,:]>2*h_min][-1]>FinalDistance:
#            break
#        front_loc = X[h[ii,:]>2*h_min][-1]
#        h_at_front = h[ii,list(np.all([X<front_loc+1,X>front_loc-1],axis=0))]
#        FrontHeight.append(np.max(h_at_front))
#        
#    return T[:ii],np.array(FrontHeight)
#
#def plotFrontHeight_vs_Time(h1init,c1init,FinalDistance,normalized = True, rootFile = 'TwoCurrentShallowWaterSimulations/SINDyData/' ,rootFileName = '',N=1000,T=30.0,h_min=0.0001,NuRe=1000,apart = 5,NuPe=None,FrSquared=2,U_s=0.01,h2init=0.0,c2init=0.0,show_legend = False, show = False, save = True, close = True,sharp = None, CFL = None):
#  
#    if not NuPe: 
#        NuPe = NuRe
#    details = buildFileName(N, T, h_min, NuRe, FrSquared, U_s, apart = apart, NuPe=NuPe, h1init=h1init, h2init=h2init, c1init=c1init, c2init=c2init,sharp = sharp, CFL = CFL)
#    t,u = FrontHeightvsTime(rootFile,rootFileName,details,FinalDistance)
#    plt.plot(t/t[-1],u,zorder=1)
#    plt.xticks([0,0.25,0.5,0.75,1.0],[0,'','','','Final Time'])
#    plt.xlabel('Final Time is when right front hits x = %i'%FinalDistance)
#    plt.ylabel('height at front')
#    plt.grid()
#    if save: plt.savefig('FrontHeight_vs_time_FinalDistance%i_Normalized%r.pdf'%(FinalDistance,normalized))
#    if show: plt.show()
#    if close: plt.close()
#
#def plotMaxVel_vs_Time(h1init,c1init,FinalDistance,normalized = True, rootFile = 'TwoCurrentShallowWaterSimulations/SINDyData/' ,rootFileName = '',N=1000,T=30.0,h_min=0.0001,NuRe=1000,apart = 5,NuPe=None,FrSquared=2,U_s=0.01,h2init=0.0,c2init=0.0,show_legend = False):
#    if not isinstance(h1init,list): 
#        h1init = [h1init]
#    if not isinstance(c1init,list): 
#        c1init = [c1init]
#  
#    NinetyNinePercent = []
#    NinetyEightPercent = []
#    NinetyFivePercent = []
#    NinetyPercent = []
#    for i,h1 in enumerate(h1init):
#        for j,c1 in enumerate(c1init):
#            if not NuPe: 
#                NuPe = NuRe
#            details = buildFileName(N, T, h_min, NuRe, FrSquared, U_s, apart = apart, NuPe=NuPe, h1init=h1, h2init=h2init, c1init=c1, c2init=c2init)
#            t,u = MaxVelvsTime(rootFile,rootFileName,details,FinalDistance)
#            uConst = u[-1]
#            uScale = uConst if normalized else 1.0
#            NinetyNinePercent.append([t[u>0.99*uConst][0]/t[-1],u[u>0.99*uConst][0]/uScale])
#            NinetyEightPercent.append([t[u>0.98*uConst][0]/t[-1],u[u>0.98*uConst][0]/uScale])
#            NinetyFivePercent.append([t[u>0.95*uConst][0]/t[-1],u[u>0.95*uConst][0]/uScale])
#            NinetyPercent.append([t[u>0.90*uConst][0]/t[-1],u[u>0.90*uConst][0]/uScale])
#            if normalized:
#                plt.plot(t/t[-1],u/u[-1],zorder=1)
#                #plt.plot(t/t[-1],u/u[-1],color = tabcolors[i%len(tabcolors)], linestyle = mpllinestyles[j%len(mpllinestyles)])
#            else:
#                plt.plot(t/t[-1],u,zorder=1)
#    NinetyNinePercent = np.array(NinetyNinePercent) 
#    NinetyEightPercent = np.array(NinetyEightPercent) 
#    NinetyFivePercent = np.array(NinetyFivePercent) 
#    NinetyPercent = np.array(NinetyPercent) 
#    plt.scatter(NinetyNinePercent[:,0],NinetyNinePercent[:,1],color='k',marker = mplmarkers[0],zorder=2)
#    plt.scatter(NinetyEightPercent[:,0],NinetyEightPercent[:,1],color='k',marker = mplmarkers[5],zorder=2)
#    plt.scatter(NinetyFivePercent[:,0],NinetyFivePercent[:,1],color='k',marker = mplmarkers[6],zorder=2)
#    plt.scatter(NinetyPercent[:,0],NinetyPercent[:,1],color='k',marker = mplmarkers[7],zorder=2)
#    plt.xticks([0,0.25,0.5,0.75,1.0],[0,'','','','Final Time'])
#    plt.xlabel('Final Time is when right front hits x = %i'%FinalDistance)
#    plt.ylabel('max veclocity (velocity at right front)')
#    plt.grid()
#    plt.savefig('MaxVel_vs_time_FinalDistance%i_Normalized%r.pdf'%(FinalDistance,normalized))
#    plt.show()
#
#def plotMaxVel_vs_Distance(h1init,c1init,FinalDistance,normalized = True, rootFile = 'TwoCurrentShallowWaterSimulations/SINDyData/' ,rootFileName = '',N=1000,T=30.0,h_min=0.0001,NuRe=1000,apart = 5,NuPe=None,FrSquared=2,U_s=0.01,h2init=0.0,c2init=0.0,sharp=None,CFL=None,show_legend = False):
#    if not isinstance(h1init,list): 
#        h1init = [h1init]
#    if not isinstance(c1init,list): 
#        c1init = [c1init]
#  
#    NinetyNinePercent = []
#    NinetyEightPercent = []
#    NinetyFivePercent = []
#    NinetyPercent = []
#    for i,h1 in enumerate(h1init):
#        for j,c1 in enumerate(c1init):
#            if not NuPe: 
#                NuPe = NuRe
#            details = buildFileName(N, T, h_min, NuRe, FrSquared, U_s, apart = apart, NuPe=NuPe, h1init=h1, h2init=h2init, c1init=c1, c2init=c2init,sharp = sharp, CFL = CFL)
#            rf,u = MaxVelvsDistance(rootFile,rootFileName,details,FinalDistance)
#            print('max front velocity is %0.8f, final front velocity is %0.8f, average of those velocities is %0.16f'%(np.max(u),u[-1],(np.max(u)+u[-1])/2))
#            print('\n (Froude Squared, constant velocity) = (%0.5f, %0.16f)\n'%(FrSquared,(np.max(u)+u[-1])/2))
#            rf = rf - rf[0]
#            uConst = u[-1]
#            uScale = uConst if normalized else 1.0
#            NinetyNinePercent.append([rf[u>0.99*uConst][0],u[u>0.99*uConst][0]/uScale])
#            NinetyEightPercent.append([rf[u>0.98*uConst][0],u[u>0.98*uConst][0]/uScale])
#            NinetyFivePercent.append([rf[u>0.95*uConst][0],u[u>0.95*uConst][0]/uScale])
#            NinetyPercent.append([rf[u>0.90*uConst][0],u[u>0.90*uConst][0]/uScale])
#            if normalized:
#                plt.plot(rf,u/u[-1],zorder=1)
#                #plt.plot(t/t[-1],u/u[-1],color = tabcolors[i%len(tabcolors)], linestyle = mpllinestyles[j%len(mpllinestyles)])
#            else:
#                plt.plot(rf,u,zorder=1)
#    NinetyNinePercent = np.array(NinetyNinePercent) 
#    NinetyEightPercent = np.array(NinetyEightPercent) 
#    NinetyFivePercent = np.array(NinetyFivePercent) 
#    NinetyPercent = np.array(NinetyPercent) 
#    plt.scatter(NinetyNinePercent[:,0],NinetyNinePercent[:,1],color='k',marker = mplmarkers[0],zorder=2)
#    plt.scatter(NinetyEightPercent[:,0],NinetyEightPercent[:,1],color='k',marker = mplmarkers[5],zorder=2)
#    plt.scatter(NinetyFivePercent[:,0],NinetyFivePercent[:,1],color='k',marker = mplmarkers[6],zorder=2)
#    plt.scatter(NinetyPercent[:,0],NinetyPercent[:,1],color='k',marker = mplmarkers[7],zorder=2)
#    #plt.xticks([0,0.25,0.5,0.75,1.0],[0,'','','','Final Time'])
#    plt.xlabel('Distance Taveled')
#    plt.ylabel('max veclocity (velocity at right front)')
#    plt.grid()
#    plt.savefig(rootFile + 'MaxVel_vs_Distance_FinalDistance%i_Normalized%r_FrFr%0.3f_N%i.pdf'%(FinalDistance,normalized,FrSquared,N))
#    #plt.show()
#    plt.close()
#
#def secant_root(x1,x2,y1,y2):
#    return (1-y1)*(x2-x1)/(y2-y1)+x1
#
#def CollVel_vs_initValues(h1init,c1init,FinalDistance,rootFile = 'TwoCurrentShallowWaterSimulations/SINDyData/' ,rootFileName = '',N=1000,T=30.0,h_min=0.0001,NuRe=1000,apart = 5,NuPe=None,FrSquared=2,U_s=0.01,h2init=0.0,c2init=0.0,show_legend = False):
#    if not isinstance(h1init,list): 
#        h1init = [h1init]
#    if not isinstance(c1init,list): 
#        c1init = [c1init]
#    scatterData = []
#    for i,h1 in enumerate(h1init):
#        for j,c1 in enumerate(c1init):
#            if not NuPe: 
#                NuPe = NuRe
#            details = buildFileName(N, T, h_min, NuRe, FrSquared, U_s, apart = apart, NuPe=NuPe, h1init=h1, h2init=h2init, c1init=c1, c2init=c2init)
#            t,u = MaxVelvsTime(rootFile,rootFileName,details,FinalDistance)
#            scatterData.append([h1,c1,u[-1]])
#            X = np.array(scatterData)
#    return X[:,0], X[:,1], X[:,2] 
#
#def Scatter3D(x,y,z,fig = None,sub=None, xlab='initial h',ylab='initial c',zlab='final max u',alpha = 1,show = True): 
#    if fig==None:
#        fig = plt.figure()
#        ax = fig.add_subplot(projection='3d')
#    else:
#        ax = fig.add_subplot(sub,projection='3d')
#    X = np.linspace(x[0],x[-1],250)
#    Y = np.linspace(y[0],y[-1],250)
#    X,Y = np.meshgrid(X,Y)
#    Z = alpha*np.sqrt(X*Y)
#    ax.scatter(x,y,z,c='k')
#    ax.plot_surface(X,Y,Z,cmap=cm['cool'],linewidth=0)
#    ax.set_xlabel(xlab,labelpad = 15)
#    ax.set_ylabel(ylab,labelpad = 15)
#    ax.set_zlabel(zlab,labelpad = 15)
#    if show: plt.show()
#
#def initialHC_vs_FinalU(FinalDistance):
#    plt.rcParams.update({"text.usetex":True})
#    h,c,u = CollVel_vs_initValues(h1init = [i/10 for i in range(1,11)], c1init=[i/10 for i in range(1,11)], FinalDistance = FinalDistance, rootFile = 'TestNonDimVelocity/NonDim_' ,rootFileName = '',N=2000,T=100.0,h_min=0.0001,NuRe=1000,apart = 0,NuPe=None,FrSquared=2,U_s=0.00,h2init=0.0,c2init=0.0,show_legend = False)
#    S = np.average(u/np.sqrt(h*c))
#    P = np.polyfit(np.sqrt(h*c),u,1)
#
#    x = np.sqrt(h*c)
#    LS = np.matmul(x,u)/np.matmul(x,x)
#    for (H,C,U) in zip(h,c,u):
#        print('initial h = %0.2f, initial c = %0.2f, final u = %0.2f, u/(hc)^(1/2) = %0.4f'%(H,C,U,U/np.sqrt(H*C)))
#    print('min u/(hc)^(1/2) = %0.4f'%(np.min(u/np.sqrt(h*c))))
#    print('u/(hc)^(1/2) = %0.4f, on average'%(S))
#    print('max u/(hc)^(1/2) = %0.4f'%(np.max(u/np.sqrt(h*c))))
#    
#    fig = plt.figure(figsize=[20,5])
#    Scatter3D(h,c,u,fig=fig,sub=141,alpha=1,show=False)
#    plt.title('$u^*=\sqrt{h_0c_0}$')
#    Scatter3D(h,c,u,fig=fig,sub=142,alpha=S,show=False)
#    plt.title('$u^*=%0.4f\sqrt{h_0c_0}$'%S)
#    Scatter3D(h,c,u,fig=fig,sub=143,alpha=LS,show=False)
#    plt.title('$u^*=%0.4f\sqrt{h_0c_0}$'%LS)
#    Scatter3D(h,c,u,fig=fig,sub=144,alpha=P[0],show=False)
#    plt.title('$u^*=%0.4f\sqrt{h_0c_0}$'%P[0])
#    plt.tight_layout()
#    plt.savefig('initialHC_vs_FinalU_FinalDistance%i.pdf'%FinalDistance)
#    plt.close()
#
#    plt.scatter(x,u,label = 'Final Max Velocity',color='k')
#    plt.plot(x,S*x,label = 'Average $\\alpha:\ %0.4f\sqrt{h_0c_0}$'%S)
#    plt.plot(x,LS*x,label = 'Least Squares: $%0.4f\sqrt{h_0c_0}$'%LS)
#    plt.plot(x,x*P[0]+P[1],label = 'Line of best fit: $%0.4f\sqrt{h_0c_0}%s%0.4f$'%(P[0],'+' if P[1]>=0 else '-',np.abs(P[1])))
#    x = np.sqrt(h*c)
#    plt.legend()
#    plt.xlabel('$\sqrt{h_0c_0}$')
#    plt.ylabel('Velocity')
#    plt.savefig('sqrtInitialHC_vs_FinalU_bestFit_FinalDistance%i.pdf'%FinalDistance)
#    plt.close()
# 
#    fig = plt.figure(figsize=[20,6])
#    plt.subplot(141)
#    plt.hist(u/x)
#    plt.title('$\\frac{u^*}{\sqrt{h_0c_0}}$')
#    plt.subplot(142)
#    plt.hist(u-S*x)
#    plt.title('Average $\\alpha$\n $u^*-%0.4f\sqrt{h_0c_0}$'%S)
#    plt.subplot(143)
#    plt.hist(u-LS*x)
#    plt.title('Least Squares\n $u^*-%0.4f\sqrt{h_0c_0}$'%LS)
#    plt.subplot(144)
#    plt.hist(u-(P[0]*x+P[1]))
#    plt.title('Line of best fit\n $u^*-(%0.4f\sqrt{h_0c_0}%s%0.4f$)'%(P[0],'+' if P[1]>=0 else '-',np.abs(P[1])))
#    plt.tight_layout()
#    #plt.subplots_adjust(left
#    plt.savefig('AlphaValueAnalsysis_InitialHC_vs_FinalU_FinalDistance%i.pdf'%FinalDistance)
#    plt.close()
#
#def make_list(x):
#    if not isinstance(x,list): return [x]
#    else: return x
#
#def plot_params(param,zoom_window='all', which='h', rootFile = 'Baseline/', rootFileName = '', N=16000, T=30.0, h_min=0.0001, NuRe=1000, CFL=0.1, sharp=200, apart = 5, NuPe=None, FrSquared=2.828, U_s=0.00, h1init=1.0, h2init=1.0, c1init=1.0, c2init=1.0, legend = [],show=False,legend_title=None,save=True):
#    plt.figure()
#    def get_ylim(x,u,a,b):
#        u=u[list(np.all([x<b,x>a],axis=0))]
#        return np.min(u),np.max(u)
#        #max_,min_ = np.max(u),np.min(u)
#        #offset = (max_-min_)*0.05
#        #return [min_-offset,max_+offset]
# 
#        
#    h_min = make_list(h_min)
#    N = make_list(N)
#    NuRe = make_list(NuRe)
#    sharp = make_list(sharp)
#    CFL = make_list(CFL)
#
#    FrSquared = make_list(FrSquared)
#    U_s = make_list(U_s)
#
#    if not NuPe: 
#        NuPe = NuRe
#    else:
#        NuPe = make_list(NuPe)
#
#    print(param + ', ' + zoom_window)
#    near_origin = [0,2]
#    near_origin_y = [100,-100]
#    front = [100,-100]
#    front_y = [100,-100]
#    bore = [100,-100]
#    bore_y = [100,-100]
#    ls_counter = 2
#    for hm in h_min:
#        for nn in N:
#            for NumRey,NumPec in zip(NuRe,NuPe):
#                for sss in sharp:
#                    for cfl in CFL:
#                        for fr2 in FrSquared:
#                            for us in U_s:
#                                print(hm,nn,NumRey,sss,cfl,fr2,us)
#                                details = buildFileName(N=nn, T=T, h_min = hm, NuRe = NumRey, FrSquared = fr2, U_s = us,sharp=sss,CFL=cfl, apart = apart, NuPe=NumPec, h1init=h1init, h2init=h2init, c1init=c1init, c2init=c2init)
#                                x,t,h = unpack(rootFile + rootFileName + 'h' + details)
#                                if which in ['u','c1','c2']:
#                                    _,_,u = unpack(rootFile + rootFileName + char_to_cons[which] + details)
#                                    u=u/h
#                                else:
#                                    _,_,u = unpack(rootFile + rootFileName + which + details)
#                                print(' \n t = %0.16f \n'%(t[-1]))
#                                front_loc = x[h[-1,:]>2*hm][-1]
#                                if param == 'SpaceDiscretization':
#                                    front = [min(front[0],front_loc-0.1),max(front[1],front_loc+0.02)]
#                                if param == 'CFL':
#                                    front = [min(front[0],front_loc-0.1),max(front[1],front_loc+0.02)]
#                                else:
#                                    front = [min(front[0],front_loc-2.5),max(front[1],front_loc+0.5)]
#                                front_y = [min(front_y[0],get_ylim(x,u[-1,:],front[0],front[1])[0]),max(front_y[1],get_ylim(x,u[-1,:],front[0],front[1])[1])]
#                                #h_at_front = h[-1,list(np.all([x<front[1],x>front[0]],axis=0))]
#                                behind_front = list(np.all([x<front_loc-1,x>0],axis=0))
#                                h_behind_front = h[-1,behind_front]
#                                x_behind_front = x[behind_front]
#                                level_cut = (np.max(h_behind_front)+h_behind_front[-1])/2.0
#                                #snowplow_max = np.max(u_at_front)
#                                #bore_loc = x[u[-1,:]>1.01*snowplow_max][-1]
#                                bore_loc = x_behind_front[h_behind_front>level_cut][-1]
#                                print(bore_loc)
#                                if param == 'SpaceDiscretization':
#                                    bore = [max(min(bore[0],bore_loc-0.02),0),max(bore[1],bore_loc+0.02)]
#                                if param == 'CFL':
#                                    bore = [max(min(bore[0],bore_loc-0.01),0),max(bore[1],bore_loc+0.01)]
#                                else:
#                                    bore = [max(min(bore[0],bore_loc-0.5),0),max(bore[1],bore_loc+0.5)]
#                                bore_y = [min(bore_y[0],get_ylim(x,u[-1,:],bore[0],bore[1])[0]),max(bore_y[1],get_ylim(x,u[-1,:],bore[0],bore[1])[1])]
#                                near_origin_y = [min(near_origin_y[0],get_ylim(x,u[-1,:],near_origin[0],near_origin[1])[0]),max(near_origin_y[1],get_ylim(x,u[-1,:],near_origin[0],near_origin[1])[1])]
#                                plt.plot(x,u[-1,:],linestyle = mpllinestyles[ls_counter%len(mpllinestyles)],alpha=0.5)
#                                #plt.plot(x,u[-1,:])
#                                ls_counter+=1
#    print(' ')
#    plt.legend(legend,title = legend_title)
#    if zoom_window == 'all':
#        plt.xlim([0,x[-1]])
#        if which == 'u':
#            plt.ylim([-0.05,1.05])
#        savelabel = ''
#    elif zoom_window == 'front':
#        #plt.ylim(front_y)
#        offset = (front_y[1]-front_y[0])*0.05
#        plt.ylim([front_y[0]-offset,front_y[1]+offset])
#        plt.xlim(front)
#        savelabel = '_near_front'
#    elif zoom_window == 'bore':
#        #plt.ylim(bore_y)
#        offset = (bore_y[1]-bore_y[0])*0.05
#        plt.ylim([bore_y[0]-offset,bore_y[1]+offset])
#        plt.xlim(bore)
#        savelabel = '_near_bore'
#    elif zoom_window == 'origin':
#        plt.xlim(near_origin)
#        offset = (near_origin_y[1]-near_origin_y[0])*0.05
#        plt.ylim([near_origin_y[0]-offset,near_origin_y[1]+offset])
#        savelabel = '_near_origin'
#    plt.xlabel('x')
#    plt.ylabel(variable_dict[which])
#    if show: plt.show()
#    if save: plt.savefig(rootFile + 'solutions/plots/' + param + '_' + which + savelabel + '.pdf')
#    #plt.close()
#
#def run_params():
#    plt.rcParams.update({"text.usetex":True})
#    for zoom_window in ['all','front','bore','origin']:
#        for which in ['u','q','h']:
#            #plot_params('Froude',zoom_window=zoom_window,FrSquared = [2,4,8],legend = ['Fr2 = %i'%f for f in [2,4,8]])
#            plot_params('SpaceDiscretization',legend_title='$N$',which=which,zoom_window=zoom_window,N=[4000,8000,16000,32000], legend = ['%i'%f for f in [4000,8000,16000,32000]])
#            plot_params('h_min',legend_title='$h_{\\textrm{min}}$',which=which,zoom_window=zoom_window,h_min = [0.00005,0.0001,0.0002,0.0004], legend = ['%0.5f'%f for f in [0.00005,0.0001,0.0002,0.0004]])
#            plot_params('NuRe',legend_title='$\\textrm{Re}, \\textrm{Pe}$' ,which=which,zoom_window=zoom_window,NuRe = [250,500,1000,2000], legend = ['%i'%f for f in [250,500,1000,2000]])
#            plot_params('sharpness',legend_title='s',which=which,zoom_window=zoom_window,sharp = [50,100,200,400], legend = ['%i'%f for f in [50,100,200,400]])
#            plot_params('CFL',legend_title='CFL',which=which,zoom_window=zoom_window,CFL = [0.4,0.2,0.1,0.05], legend = ['%0.2f'%f for f in [0.4,0.2,0.1,0.05]])
#            plot_params('U_s',legend_title='$u_s$' ,which=which,zoom_window=zoom_window,U_s = [0.0,0.01,0.02],legend = ['%0.2f'%f for f in [0.0,0.01,0.02]])
#    for zoom_window in ['all','front']:
#        for which in ['c1','c2']:
#            #plot_params('Froude',zoom_window=zoom_window,FrSquared = [2,4,8],legend = ['Fr2 = %i'%f for f in [2,4,8]])
#            plot_params('SpaceDiscretization',legend_title='$N$',which=which,zoom_window=zoom_window,N=[4000,8000,16000,32000], legend = ['%i'%f for f in [4000,8000,16000,32000]])
#            plot_params('h_min',legend_title='$h_{\\textrm{min}}$',which=which,zoom_window=zoom_window,h_min = [0.00005,0.0001,0.0002,0.0004], legend = ['%0.5f'%f for f in [0.00005,0.0001,0.0002,0.0004]])
#            plot_params('NuRe',legend_title='$\\textrm{Re}, \\textrm{Pe}$',which=which,zoom_window=zoom_window,NuRe = [250,500,1000,2000], legend = ['%i'%f for f in [250,500,1000,2000]])
#            plot_params('sharpness',legend_title='s',which=which,zoom_window=zoom_window,sharp = [50,100,200,400], legend = ['%i'%f for f in [50,100,200,400]])
#            plot_params('CFL',legend_title='CFL',which=which,zoom_window=zoom_window,CFL = [0.4,0.2,0.1,0.05], legend = ['%0.2f'%f for f in [0.4,0.2,0.1,0.05]])
#            plot_params('U_s',legend_title='$u_s$',which=which,zoom_window=zoom_window,U_s = [0.0,0.01,0.02],legend = ['%0.2f'%f for f in [0.0,0.01,0.02]])
#    plt.rcParams.update({"text.usetex":False})
#
#def Onesided_plots(whichfile, upper_x=5.0,c1init=1.0,c2init=1.0,h1init=1.0,h2init=1.0, rootFile='RH/',save = True, show = False, Nt=12,Ts=0, n =3, show_legend = False,N=16000,apart=5,CFL=0.1,T=40.0,NuRe=1000,FrSquared=2.828,U_s=0.0,h_min=0.0001,sharp=50):
#    plt.figure(figsize=[9,4])
#    example_timeplots(rootFile = rootFile,whichfile = whichfile,n=n,Nt = Nt,Ts=Ts,c1init = c1init, c2init = c2init, h1init = h1init, h2init = h2init, linestyle = 'solid',show_legend = show_legend,N=N,apart=apart,CFL=CFL,T=T,NuRe=NuRe,FrSquared=FrSquared,U_s=U_s,h_min=h_min,sharp=sharp)
#
#    plt.xlim([-0,upper_x])
#    if save:
#        filename = 'front_schematic_%s_cOne%0.1f_cTwo%0.1f_hOne%0.1f_hTwo%0.1f_upperx_%i'%(whichfile,c1init,c2init,h1init,h2init,upper_x)
#        plt.savefig(rootFile + 'solutions/plots/' + filename.replace('.','_') + '.pdf')
#        if not save: plt.close()
#    if show: plt.show()
