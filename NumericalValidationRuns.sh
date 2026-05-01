#!/bin/bash                                                                                                                                                                     
gcc -O3 -march=native -o num_val_Apr29.out TwoCurrSWEsolver.c # Compile main

# # Iterate through numbers 1 to Nruns
# for (( i=70; i<=$Nmax; i+=10 )) # Does it do endpoint??                                                                                                                            
# do
# the ampersand is the magic that runs it in parallel
# inputs go in this order N Reynolds CFL h_min sharp U_s c2init h2init
U_s_list="0.0 0.02"
N_list="7000 14000 28000 56000"
s_list="50. 100. 200. 400."
CFL_list="0.4 0.2 0.1 0.05"
Re_list="250 500 1000 2000"
h_list="0.0004 0.0002 0.0001 0.00005"

#for u_s in $U_s_list; do
#  for s in $s_list; do
#    #mkdir "outputs/1_1_${runind}_outputs/" 
#    ./num_val_Apr29.out 20000 1000 0.1 0.0001 $s $u_s 1.0 1.0 &
#  done
#done

#wait

for u_s in $U_s_list; do
  for Re in $Re_list; do
    #mkdir "outputs/1_1_${runind}_outputs/" 
    ./num_val_Apr29.out 28000 $Re 0.1 0.0001 200 $u_s 1.0 1.0 &
  done
done

wait

for u_s in $U_s_list; do
  for h in $h_list; do
    #mkdir "outputs/1_1_${runind}_outputs/" 
    ./num_val_Apr29.out 28000 1000 0.1 $h 200 $u_s 1.0 1.0 &
  done
done
# done

wait

for u_s in $U_s_list; do
  for N in $N_list; do
    #mkdir "outputs/1_1_${runind}_outputs/" 
    ./num_val_Apr29.out $N 1000 0.1 0.0001 200 $u_s 1.0 1.0 &
  done
done
# done

wait

for u_s in $U_s_list; do
  for CFL in $CFL_list; do
    #mkdir "outputs/1_1_${runind}_outputs/" 
    ./num_val_Apr29.out 28000 1000 $CFL 0.0001 200 $u_s 1.0 1.0 &
  done
done
# done

wait

