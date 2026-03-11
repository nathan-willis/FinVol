#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<sys/stat.h>
#include<time.h>
#include<string.h>
//#include "TwoCurrTestValues.h"
//#include "utility_fns.h"

// Numerical parameters 
//#define N 8000 // N+1 nodes, N cells
#define m 3  // WENO stencil size
//#define NuRe 1000. // Numerical Reynolds number
//#define NuPe NuRe // Numerical Peclet number
//#define h_min 0.0001 // minimum thickness
//#define CFL 0.1 // CFL number Delta_t/Delta_x
//#define sharp 50. // Sharpness parameter for initial conditions

int N;
double NuRe, NuPe, CFL, h_min, sharp;
double c2init, h2init;

// Physical parameters
#define a -70. // lower bound of the interval
#define b 70.  // upper bound of the interval
#define T 40. // Final Time
#define FrSquared 1.0 // Froude number
//#define U_s 0.00 // Settling speed
//double U_s = 0.0;

double U_s;

// Initital conditions parameters 
#define apart 5.0 // How far apart are the centers of the current
#define h1init 1.0
//#define h2init 1.0
//double h2init = 0.7;
#define c1init 1.0
//double c1init = 0.7;
//#define c2init 1.0
//double c2init = 0.7;
#define cur1wid 1.0
#define cur2wid 1.0

// File information
#define fileprefix "Mar3_DepositionExamplePlots/"
#define subfile "sims/"

int save_q = 0; //Decide if you want to save to a file or not.
int save_h = 0; //Decide if you want to save to a file or not.
int save_phi1 = 0; //Decide if you want to save to a file or not.
int save_phi2 = 0; //Decide if you want to save to a file or not.
int save_deposit = 1; //Decide if you want to save to a file or not.
int J_save = 1; // jump between spatial cells that are saved.  
int test_ = 0; // Do you want to compare to the values in TwoCurrTestValues.h?
double print_when = 0.01; // Save timestamp this often
double print_check = 0.0; // Check if you should save timestamp
double print_to_screen = .5; // Print to screen this often
double print_screen_check = 0.0; // Check if you should print to screen 
int print_info = 1;
double print_first_line;

int collision = 0; // Flag to know if a collision was detected or not. 
double CT, CI, CX; // declaring variables to store collsion time, index, and position. 
int is_last = 0; // Check if this is the last iteration. 

double time_check; // Declare a variable to store the time to time the finite volume solver

double *x_;
double *x;
double *dx; 

double *h_; // Declaring pointer to vector of nodal points for h
double *h; // Declaring pointer to vector of cell averages for h
double *h_temp1;// Declaring pointer to temporary h vector for RK3
double *h_temp2;// Declaring pointer to temporary h vector for RK3
double *hL;// Declaring pointer to left h vector for WENO and input of Lax-Friedrichs flux
double *hR;// Declaring pointer to right h vector for WENO and input of Lax-Friedrichs flux
// Doing all the same declarations above, but now for q, which is hu or (height)(velocity)
double *q_,*q,*q_temp1,*q_temp2,*qL,*qR;
// Doing all the same declarations above, but now for phi1, which is (height)(concentration 1)
double *phi1_,*phi1,*phi1_temp1,*phi1_temp2,*phi1L,*phi1R;
// Doing all the same declarations above, but now for phi2, which is (height)(concentration 2)
double *phi2_,*phi2,*phi2_temp1,*phi2_temp2,*phi2L,*phi2R;
double *deposit1,*deposit2;

FILE *h_file,*q_file,*phi1_file,*phi2_file,*deposit1_file,*deposit2_file,*log_file; // Variable identifying a file


double poly_spline(double x, double M, double h, double c, double elseval);
double squarewave(double x, double center, double width, double max, double min);
void initialize();
void avg_cell(double *u, double *u_avg);
void print_to_file(double t, double *x,FILE *hist2);
void print_conserved_variables_to_file(double t);
void print_params_to_log();
void print_date_time();
void pointers ();
void unpointers ();
void print_usage(const char *prog);

// These are functions called in the while loop. 
//int BC(int aa);
static inline int BC(int aa){
    if(aa<0){return BC(aa+N);}
    else{return aa%N;}
}
//double diff(double *u,int i,int which);
static inline double diff(double *u,int i,int which){
    double diff_const;
    double d_co[4] = {-49./18.,3./2.,-3./20.,1./90.}; // coefficients for 6th-order central finite difference stencil for second-derivatve

    if(which == 0){diff_const = 1./NuRe;}
    if(which == 1){diff_const = 1./NuPe;}
    return diff_const*(1./(*dx*(*dx)))*(u[BC(i-3)]*d_co[3] + u[BC(i-2)]*d_co[2] + u[BC(i-1)]*d_co[1] + u[i]*d_co[0] + u[BC(i+1)]*d_co[1] + u[BC(i+2)]*d_co[2] + u[BC(i+3)]*d_co[3]);
}
//double flux(double h, double q, double phi1, double phi2, int which);
static inline double flux(double h, double q, double phi1, double phi2, int which){
    if(which == 0){return q ;}
    if(which == 1){return q*q/h + h*phi1/(2*FrSquared) + h*phi2/(2*FrSquared) ;}
    if(which == 2){return phi1*q/h ;}
    if(which == 3){return phi2*q/h ;}
    printf("Did something weird in flux");
    return 10000000.;
    }
//double F_HLL(double h_l, double h_r, double q_l, double q_r, double phi1_l, double phi1_r, double phi2_l, double phi2_r, int which);
static inline double F_HLL(double h_l, double h_r, double q_l, double q_r, double phi1_l, double phi1_r, double phi2_l, double phi2_r, int which){
    double sp;
    double sm;
    double whichVar[4] = {h_r-h_l,q_r-q_l,phi1_r-phi1_l,phi2_r-phi2_l};
    sp =         q_r/h_r + pow((phi1_r+phi2_r)/FrSquared,1./2.);
    sp = fmax(sp,q_r/h_r);
    sp = fmax(sp,q_r/h_r - pow((phi1_r+phi2_r)/FrSquared,1./2.));
    sp = fmax(sp,q_l/h_l + pow((phi1_l+phi2_l)/FrSquared,1./2.));
    sp = fmax(sp,q_l/h_l);
    sp = fmax(sp,q_l/h_l - pow((phi1_l+phi2_l)/FrSquared,1./2.));
    sm =         q_l/h_l + pow((phi1_l+phi2_l)/FrSquared,1./2.);
    sm = fmin(sm,q_l/h_l);
    sm = fmin(sm,q_l/h_l - pow((phi1_l+phi2_l)/FrSquared,1./2.));
    sm = fmin(sm,q_r/h_r + pow((phi1_r+phi2_r)/FrSquared,1./2.));
    sm = fmin(sm,q_r/h_r);
    sm = fmin(sm,q_r/h_r - pow((phi1_r+phi2_r)/FrSquared,1./2.));

    if(sm >=0.){return flux(h_l,q_l,phi1_l,phi2_l,which);}
    if(sp <=0.){return flux(h_r,q_r,phi1_r,phi2_r,which);}
    return (sp*flux(h_l,q_l,phi1_l,phi2_l,which) - sm*flux(h_r,q_r,phi1_r,phi2_r,which) + sp*sm*whichVar[which])/(sp-sm);
    printf("Did something weird in F_HLL");
    return 10000000.;
}
//double DiscreteSpatial(int i, int which);
static inline double DiscreteSpatial(int i, int which){
    return F_HLL(hL[i],hR[BC(i+1)],qL[i],qR[BC(i+1)],phi1L[i],phi1R[BC(i+1)],phi2L[i],phi2R[BC(i+1)],which) - 
           F_HLL(hL[BC(i-1)],hR[i],qL[BC(i-1)],qR[i],phi1L[BC(i-1)],phi1R[i],phi2L[BC(i-1)],phi2R[i],which);
}
//void WENO(double *u_, double *u_left,double *u_right);
static inline void WENO(double *u_, double *u_left,double *u_right){
    // WENO constants
    double beta[2][3] = {{3./10.,3./5.,1./10.},{1./10.,3./5.,3./10.}};
    double C[4][3] = {{11./6.,-7./6.,1./3.},{1./3.,5./6.,-1./6.},{-1./6.,5./6.,1./3.},{1./3.,-7./6.,11./6.}};
    double ep = 1e-6; 
    //Need to create u_left and u_right outside of this function
    double gamma[N][m], u_l[N][m], u_r[N][m];
    double alpha_l[N][m], alpha_r[N][m];
    double W_l[N][m], W_r[N][m];
    int i,r;
    for(i=0;i<N;i++){
        gamma[i][0] = 13./12.*pow(u_[i] - 2.*u_[BC(i+1)] + u_[BC(i+2)],2) + 1./4.*pow(3.*u_[i] - 4.*u_[BC(i+1)] + u_[BC(i+2)],2);
        gamma[i][1] = 13./12.*pow(u_[BC(i-1)] - 2.*u_[i] + u_[BC(i+1)],2) + 1./4.*pow(u_[BC(i-1)] - u_[BC(i+1)],2);
        gamma[i][2] = 13./12.*pow(u_[BC(i-2)] - 2.*u_[BC(i-1)] + u_[i],2) + 1./4.*pow(u_[BC(i-2)] - 4.*u_[BC(i-1)] + 3.*u_[i],2);

        for(r=0;r<m;r++){
            u_l[i][r] = C[r+1][0]*u_[BC(i-r)] + C[r+1][1]*u_[BC(i+1-r)] + C[r+1][2]*u_[BC(i+2-r)];
            u_r[i][r] = C[r][0]*u_[BC(i-r)]   + C[r][1]*u_[BC(i+1-r)]   + C[r][2]*u_[BC(i+2-r)];

            alpha_l[i][r] = beta[0][r]/pow(ep + gamma[i][r],2);
            alpha_r[i][r] = beta[1][r]/pow(ep + gamma[i][r],2);
        }
    }

    double alpha_l_row_sum, alpha_r_row_sum;
    for(i=0;i<N;i++){
        alpha_l_row_sum = 0;
        alpha_r_row_sum = 0;
        for(r=0;r<m;r++){
            alpha_l_row_sum += alpha_l[i][r];
            alpha_r_row_sum += alpha_r[i][r];
        }
        for(r=0;r<m;r++){
            W_l[i][r] = alpha_l[i][r]/alpha_l_row_sum;
            W_r[i][r] = alpha_r[i][r]/alpha_r_row_sum;
        }
    }
   
    for(i=0;i<N;i++){
        u_left[i] = 0;
        u_right[i] = 0;
        for(r=0;r<m;r++){
            u_left[i] += W_l[i][r]*u_l[i][r];
            u_right[i] += W_r[i][r]*u_r[i][r];
        }
    }
}
//void run_WENO(double *hh, double *qq, double *pphi1, double *pphi2);
static inline void run_WENO(double *hh, double *qq, double *pphi1, double *pphi2){
    WENO(hh,hL,hR); // Perform WENO to get uL and uR from the last solution vector of cell averages
    WENO(qq,qL,qR); // Perform WENO to get uL and uR from the last solution vector of cell averages
    WENO(pphi1,phi1L,phi1R); // Perform WENO to get uL and uR from the last solution vector of cell averages
    WENO(pphi2,phi2L,phi2R); // Perform WENO to get uL and uR from the last solution vector of cell averages
}
//int max_fct(int aa, int bb);
static inline int max_fct(int aa, int bb){
    if(aa>bb){return aa;}
    else{return bb;}
}
//int min_fct(int aa, int bb);
static inline int min_fct(int aa, int bb){
    if(aa>bb){return bb;}
    else{return aa;}
}
//void collision_check();
static inline void collision_check(){
    int i;
    int phi1_right = 0;
    int phi2_left = N;
    double phi_tol = 0.05;
    for(i=0;i<N;i++){
        if(phi1[i]>phi_tol){
            phi1_right = max_fct(i,phi1_right);
        }
        if(phi2[i]>phi_tol){
            phi2_left = min_fct(i,phi2_left);
        }
    }
    if(phi1_right > phi2_left){
        collision = 1;
        CI = ((double)phi1_right + (double)phi2_left)/2.;
        CX = (x[phi1_right] + x[phi2_left])/2.;
    }
}

int main(int argc, char* argv[]){
    if (argc == 1 ||
        (argc > 1 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")))) {
        print_usage(argv[0]);
        return 0;
    }
    if (argc < 9) {
        fprintf(stderr, "Error: missing arguments.\n");
        fprintf(stderr, "Usage: %s <input_file> [other options]\n", argv[0]);
        return EXIT_FAILURE;
    }
    //double c1init = ((double)atoi(argv[1]))/100.0;
    //double U_s = ((double)atoi(argv[1]))/1000.0;
    // N NuRe CFL h_min sharp U_s c2init h2init
    N = ((int)atoi(argv[1]));
    NuRe = ((double)atoi(argv[2]));
    CFL = ((double)atof(argv[3]));
    h_min = ((double)atof(argv[4]));
    sharp = ((double)atof(argv[5]));
    U_s = ((double)atof(argv[6]));
    c2init = ((double)atof(argv[7]));
    h2init = ((double)atof(argv[8]));
    NuPe = NuRe;

    is_last = 0;
    //print_when = 1.0;
    print_check = 0.0;
    time_check = clock();
    pointers();
    
    char file_details[256];
    char dir[512], str_log[1024]; // String of characters containing directory name and log name
    char str_h[1024],str_q[1024],str_phi1[1024],str_phi2[1024],str_deposit1[1024],str_deposit2[1024]; // String of characters containing file name
    sprintf(file_details, "hTwo%0.2f_cTwo%0.2f_%1.0fapart_N%i_CFL%0.3f_T%0.1f_NuRe%0.0f_Us%0.3f_hmin%0.5f_sharp%0.0f", h2init,c2init,apart,N,CFL,T,NuRe,U_s,h_min,sharp); // Building a string that contains all the parameter details for the file names
    sprintf(dir, "%s%s%s",fileprefix,subfile,file_details); // Building the directory name
    sprintf(str_h, "%s/h", dir); // Building a complete file name
    sprintf(str_q, "%s/q", dir); // Building a complete file name
    sprintf(str_phi1, "%s/phi1", dir); // Building a complete file name
    sprintf(str_phi2, "%s/phi2", dir); // Building a complete file name
    sprintf(str_log, "%s/info.log", dir); // Building a file name for the log
    sprintf(str_deposit1, "%s/d1", dir); // Building a complete file name
    sprintf(str_deposit2, "%s/d2", dir); // Building a complete file name

    initialize(); // 
    
    //double k = pow(*dx,5./3.)/2; // time step to satisfy CFL-condtion with RK2 time stepping
    double k = (*dx)*CFL;
    double t = 0.; // initialize time

    if(save_deposit || save_q || save_h || save_phi1 || save_phi2){
        FILE *tempfile;
        if ((tempfile = fopen(str_log, "r")))
        {
            // Checking if the files already exist before opening in append mode. 
            printf("WHOA! The files already exists. This will append and mess up old data. \n Exiting now.");
            fclose(tempfile);
            exit(-1);
        }
        mkdir(dir,0777); // Create a directory to store solutions and log file (the 0777 is permission allowing anyone to read, write, or edit the directory)
        // Open all the files in append mode
        log_file  = fopen(str_log,"a"); // open log file as one that can append (the "a")
        if(save_h){h_file    = fopen(str_h,"a");} // open file as one that can append (the "a")
        if(save_q){q_file    = fopen(str_q,"a");} // open file as one that can append (the "a")
        if(save_phi1){phi1_file = fopen(str_phi1,"a");} // open file as one that can append (the "a")
        if(save_phi2){phi2_file = fopen(str_phi2,"a");} // open file as one that can append (the "a")
        if(save_deposit){deposit1_file    = fopen(str_deposit1,"a");} // open file as one that can append (the "a")
        if(save_deposit){deposit2_file    = fopen(str_deposit2,"a");} // open file as one that can append (the "a")

        print_conserved_variables_to_file(t);
        print_params_to_log();
    }

    int i;
    printf("\n##########################################\n");
    printf("# c1=%0.2f,  c2=%0.2f,  h1=%.2f,  h2=%0.2f \n# N=%i,  dx=%0.8f\n# CFL=%0.4f,  k=%0.8f\n# t=%0.8f\n# Reynolds = %4.0f, U_s = %0.3f\n",c1init,c2init,h1init,h2init,N,*dx,k/(*dx),k,t,NuRe,U_s);
    printf("##########################################\n\n");
    while(t<T){
        if(t+k>T){
            k = T-t;
            is_last = 1;
        }
        run_WENO(h,q,phi1,phi2);
        for(i=0;i<N;i++){
            h_temp1[i]    = h[i]    - k/(*dx)*DiscreteSpatial(i,0) + k*diff(h,i,0); // First step of RK2
            q_temp1[i]    = q[i]    - k/(*dx)*DiscreteSpatial(i,1) + k*diff(q,i,0);// First step of RK2
            phi1_temp1[i] = phi1[i] - k*(U_s*phi1[i]/h[i] + DiscreteSpatial(i,2)/(*dx)) + k*diff(phi1,i,1);// First step of RK2
            phi2_temp1[i] = phi2[i] - k*(U_s*phi2[i]/h[i] + DiscreteSpatial(i,3)/(*dx)) + k*diff(phi2,i,1);// First step of RK2
        }
        run_WENO(h_temp1,q_temp1,phi1_temp1,phi2_temp1);
        for(i=0;i<N;i++){
            h_temp2[i]    = 3./4.*h[i]    + 1./4.*h_temp1[i]    - k/(4.*(*dx))*DiscreteSpatial(i,0) + (1./4.)*k*diff(h_temp1,i,0);
            q_temp2[i]    = 3./4.*q[i]    + 1./4.*q_temp1[i]    - k/(4.*(*dx))*DiscreteSpatial(i,1) + (1./4.)*k*diff(q_temp1,i,0);
            phi1_temp2[i] = 3./4.*phi1[i] + 1./4.*phi1_temp1[i] - k/4.*(U_s*phi1_temp1[i]/h[i] + DiscreteSpatial(i,2)/(*dx)) + (1./4.)*k*diff(phi1_temp1,i,1);
            phi2_temp2[i] = 3./4.*phi2[i] + 1./4.*phi2_temp1[i] - k/4.*(U_s*phi2_temp1[i]/h[i] + DiscreteSpatial(i,3)/(*dx)) + (1./4.)*k*diff(phi2_temp1,i,1);
        }
        run_WENO(h_temp2,q_temp2,phi1_temp2,phi2_temp2);
        for(i=0;i<N;i++){
            h[i]    = h[i]/3.    + 2./3.*h_temp2[i]    - 2.*k/(3.*(*dx))*DiscreteSpatial(i,0) + (2./3.)*k*diff(h_temp2,i,0); // Second step of RK2
            q[i]    = q[i]/3.    + 2./3.*q_temp2[i]    - 2.*k/(3.*(*dx))*DiscreteSpatial(i,1) + (2./3.)*k*diff(q_temp2,i,0); // Second step of RK2
            phi1[i] = phi1[i]/3. + 2./3.*phi1_temp2[i] - 2.*k/3.*(U_s*phi1_temp2[i]/h[i] + DiscreteSpatial(i,2)/(*dx)) + (2./3.)*k*diff(phi1_temp2,i,1); // Second step of RK2
            phi2[i] = phi2[i]/3. + 2./3.*phi2_temp2[i] - 2.*k/3.*(U_s*phi2_temp2[i]/h[i] + DiscreteSpatial(i,3)/(*dx)) + (2./3.)*k*diff(phi2_temp2,i,1); // Second step of RK2
        }
        if(!collision){
            CT = t; 
            collision_check();
        }
        
        for(i=0;i<N;i++){
            deposit1[i]+=k*U_s*phi1[i]/h[i];
            deposit2[i]+=k*U_s*phi2[i]/h[i];
        }
       
        t+=k;
        print_check += k;
        print_screen_check += k;
        if(print_check>=print_when || is_last){
            if(save_deposit || save_q || save_h || save_phi1 || save_phi2){print_conserved_variables_to_file(t);
                if(print_info){printf("SAVING TO FILE: ");}
            }
            if(print_info){printf("t=%0.4f \n",t);}
            print_check = print_check - print_when;
        }
        if(print_screen_check>=print_to_screen && print_info){
            printf("\n##########################################\n");
            printf("# c1=%0.2f,  c2=%0.2f,  h1=%.2f,  h2=%0.2f \n# N=%i,  dx=%0.8f\n# CFL=%0.4f,  k=%0.16f\n# t=%0.16f\n",c1init,c2init,h1init,h2init,N,*dx,CFL,k,t);
            printf("##########################################\n\n");
            print_screen_check -= print_to_screen;
        }
        if(isnan(h[N/2])){
           printf("\n \n WAH! BLOW UP!! \n \n");
           fprintf(log_file,"\nBLOW UP! nan was detected at the center of the spatial domain and code was terminated at time t=%0.4f.\n",t);
           t = 10000000;
        }
    }

    /*if(test_){
    int checker = 1;
    double toler = 0.0000000000001;
    for(i=0;i<N;i++){
        if(fabs(h[i]-h_true[i])>toler){checker = 0;}
        if(fabs(q[i]-q_true[i])>toler){checker = 0;}
        if(fabs(phi1[i]-phi1_true[i])>toler){checker = 0;}
        if(fabs(phi2[i]-phi2_true[i])>toler){checker = 0;}
    }
    if(checker == 0){printf("the test FAILED\n");}
    else{printf("the test PASSED\n");}
    }*/
    unpointers();
    double runTime = (clock()-time_check)/CLOCKS_PER_SEC;
    printf("%0.8f",runTime);
    if(save_deposit || save_q || save_h || save_phi1 || save_phi2){
        fprintf(log_file,"\nCollision detected: %i", collision);
        if(collision){
            fprintf(log_file,"\ncollision time: CT=%0.16f",CT);
            fprintf(log_file,"\ncollision index: CI=%5.1f",CI);
            fprintf(log_file,"\ncollision position: CX=%0.16f",CX);
        }
        fprintf(log_file,"\n\nRun time was %.4f seconds.",runTime);
        fclose(log_file);
    }
    if(save_h){fclose(h_file);}
    if(save_q){fclose(q_file);}
    if(save_phi1){fclose(phi1_file);}
    if(save_phi2){fclose(phi2_file);}
    if(save_deposit){fclose(deposit1_file);}
    if(save_deposit){fclose(deposit2_file);}
}

double squarewave(double x, double center, double width, double max, double min){
    //double s = 50.0;
    return (max-min)*(tanh(sharp*(x-(center-width/2.))) + tanh(-sharp*(x-(center+width/2.))))/2. + min;
}

void initialize(){
    print_first_line = 1;
    int i;
    for(i=0;i<N+1;i++){
        x_[i] = (b-a)*(double)i/((double)N)+a; // linspace
    }
    *dx = x_[1]-x_[0];
    for(i=0;i<N;i++){
        x[i] = x_[i]+*dx/2.; //Shift to the midpoint of each cell
        deposit1[i]=0.;
        deposit2[i]=0.;
    }

    for(i=0;i<N+1;i++){
        q_[i]    = 0.;
        h_[i]    = squarewave(x_[i],-apart/2.0,cur1wid,fmax(0.,h1init-h_min),0.) + squarewave(x_[i],apart/2.0,cur2wid,fmax(0.,h2init-h_min),0.) + h_min;
        phi1_[i] = squarewave(x_[i],-apart/2.0,cur1wid,c1init,0.)*h_[i];
        phi2_[i] = squarewave(x_[i],apart/2.0,cur2wid,c2init,0.)*h_[i];
        //phi1_[i] = c1init*h_[i];
        //phi2_[i] = c2init*h_[i];
         
    }
    avg_cell(h_,h); // Get cell averages
    avg_cell(q_,q); // Get cell averages
    avg_cell(phi1_,phi1); // Get cell averages
    avg_cell(phi2_,phi2); // Get cell averages
}

void avg_cell(double *u, double *u_avg){
    int i;
    for(i=0;i<N;i++){
        u_avg[i] = (u[i] + u[i+1])/2.;
    }
}


void print_to_file(double t, double *x,FILE *hist2){
    fprintf(hist2,"%3.10f ",t);
    int i;
    for(i=0;i<N;i++){
        if((i+J_save/2)%J_save==0){fprintf(hist2,"%3.10f ",x[i]);}
    }
    fprintf(hist2,"\n");
}

void print_conserved_variables_to_file(double t){
    // Save each of the discretized space vector to each file
    // Then save the initial conditions
    // It is important that x gets printed before each IC, b/c the post proc will unpack the .npy files in that way. 
    if(print_first_line){
        if(save_phi1){print_to_file(t,x,phi1_file);}
        if(save_phi2){print_to_file(t,x,phi2_file);}
        if(save_q){print_to_file(t,x,q_file);}
        if(save_h){print_to_file(t,x,h_file);}
        if(save_deposit){print_to_file(t,x,deposit1_file);}
        if(save_deposit){print_to_file(t,x,deposit2_file);}
        print_first_line=0;
    }
    if(save_phi1){print_to_file(t,phi1,phi1_file);}
    if(save_phi2){print_to_file(t,phi2,phi2_file);}
    if(save_q){print_to_file(t,q,q_file);}
    if(save_h){print_to_file(t,h,h_file);}
    if(save_deposit){print_to_file(t,deposit1,deposit1_file);}
    if(save_deposit){print_to_file(t,deposit2,deposit2_file);}
}

void print_params_to_log(){
    print_date_time();
    // Numerical parameters 
    fprintf(log_file,"    Numerical Parameters \n \n");
    fprintf(log_file,"N=%i, N+1 node, N cells\n",N);
    fprintf(log_file,"Numerical Reynolds = %4.0f\n",NuRe);
    fprintf(log_file,"Numerical Peclet = %4.0f\n",NuPe);
    fprintf(log_file,"h_min = %0.5f, minimum height for SWE\n",h_min);
    fprintf(log_file,"CFL = %0.5f\n",CFL);
    
    // Physical parameters
    fprintf(log_file,"\n    Physical Parameters \n \n");
    fprintf(log_file,"Domain size: [%0.1f,%0.1f]\n",a,b);
    fprintf(log_file,"T = %0.2f, Final Time\n",T);
    fprintf(log_file,"Fr2 = %0.4f, Froude number squared\n",FrSquared);
    fprintf(log_file,"U_s = %0.3f, settling speed\n",U_s);
    
    
    // Initital conditions parameters 
    fprintf(log_file,"\n    Initial Conditions Parameters \n \n");
    fprintf(log_file,"h1init = %0.2f, initial height of left current\n",h1init);
    fprintf(log_file,"c1init = %0.2f, initial concentration of left current\n",c1init);
    fprintf(log_file,"cur1wid = %0.2f, initial width of left current\n",c1init);
    fprintf(log_file,"h2init = %0.2f, initial height of right current\n",h2init);
    fprintf(log_file,"c2init = %0.2f, initial concentration of right current\n",c2init);
    fprintf(log_file,"cur2wid = %0.2f, initial width of right current\n",c1init);
    fprintf(log_file,"apart = %0.2f, initial distance between centers of the currents\n",apart);
    fprintf(log_file,"sharp = %4.0f, sharpness parameter for hyperbolic tangent\n",sharp);
}

void print_date_time(){
    // this is a function I copied from Wikipedia to print the date and time. 
    // https://en.wikipedia.org/wiki/C_date_and_time_functions
    time_t current_time;
    char* c_time_string;

    /* Obtain current time. */
    current_time = time(NULL);

    if (current_time == ((time_t)-1)){fprintf(log_file, "Failure to obtain the current time.\n");}
    else{
        c_time_string = ctime(&current_time);// Convert to local time format.
        if (c_time_string == NULL){fprintf(log_file, "Failure to convert the current time.\n");}
        else{fprintf(log_file,"Current time is %s\n", c_time_string);}// Print to stdout. ctime() has already added a terminating newline character.
    }
}

void pointers (){
    x_= malloc((N+1)*sizeof(double));
    x = malloc((N)*sizeof(double));
    dx = malloc(sizeof(double)); 
    
    h_= malloc((N+1)*sizeof(double)); // Declaring pointer to vector of nodal points for h
    h = malloc((N)*sizeof(double)); // Declaring pointer to vector of cell averages for h
    h_temp1 = malloc((N)*sizeof(double));// Declaring pointer to temporary h vector for RK3
    h_temp2 = malloc((N)*sizeof(double));// Declaring pointer to temporary h vector for RK3
    hL = malloc((N)*sizeof(double));// Declaring pointer to left h vector for WENO and input of Lax-Friedrichs flux
    hR = malloc((N)*sizeof(double));// Declaring pointer to right h vector for WENO and input of Lax-Friedrichs flux
    // Doing all the same declarations above, but now for q, which is hu or (height)(velocity)
    q_= malloc((N+1)*sizeof(double));
    q = malloc((N)*sizeof(double));
    q_temp1 = malloc((N)*sizeof(double));
    q_temp2 = malloc((N)*sizeof(double));
    qL = malloc((N)*sizeof(double));
    qR = malloc((N)*sizeof(double));
    // Doing all the same declarations above, but now for phi1, which is (height)(concentration 1)
    phi1_= malloc((N+1)*sizeof(double));
    phi1 = malloc((N)*sizeof(double));
    phi1_temp1 = malloc((N)*sizeof(double));
    phi1_temp2 = malloc((N)*sizeof(double));
    phi1L = malloc((N)*sizeof(double));
    phi1R = malloc((N)*sizeof(double));
    // Doing all the same declarations above, but now for phi2, which is (height)(concentration 2)
    phi2_= malloc((N+1)*sizeof(double));
    phi2 = malloc((N)*sizeof(double));
    phi2_temp1 = malloc((N)*sizeof(double));
    phi2_temp2 = malloc((N)*sizeof(double));
    phi2L = malloc((N)*sizeof(double));
    phi2R = malloc((N)*sizeof(double));

    deposit1 = malloc((N)*sizeof(double)); // Declaring pointer to vector of cell averages for deposit1
    deposit2 = malloc((N)*sizeof(double)); // Declaring pointer to vector of cell averages for deposit2
}


void unpointers (){
    //deallocate memory
    free(x); free(x_); free(dx);
    free(h); free(h_); free(h_temp1); free(h_temp2); free(hL); free(hR);
    free(q); free(q_); free(q_temp1); free(q_temp2); free(qL); free(qR);
    free(phi1); free(phi1_); free(phi1_temp1); free(phi1_temp2); free(phi1L); free(phi1R);
    free(phi2); free(phi2_); free(phi2_temp1); free(phi2_temp2); free(phi2L); free(phi2R);
    free(deposit1); free(deposit2);
}

void print_usage(const char *prog) {
    //N = ((int)atoi(argv[1]));
    //NuRe = ((double)atoi(argv[2]));
    //CFL = ((double)atof(argv[3]));
    //h_min = ((double)atof(argv[4]));
    //sharp = ((double)atof(argv[5]));
    //U_s = ((double)atof(argv[6]));
    //c2init = ((double)atof(argv[7]));
    //h2init = ((double)atof(argv[8]));
    printf("\nUsage:\n");
    printf("  %s arg1 arg2 arg3 arg4 arg5 arg6 arg7 arg8\n\n", prog);
    printf("Arguments:\n");
    printf("  arg1 N, number of cells \n");
    printf("  arg2 NuRe, numerical Reynolds number \n");
    printf("  arg3 CFL \n");
    printf("  arg4 h_min \n");
    printf("  arg5 sharp \n");
    printf("  arg6 U_s \n");
    printf("  arg7 c2init \n");
    printf("  arg8 h2init \n\n");
    printf("Options:\n");
    printf("  -h, --help   Show this help message\n\n");
}
