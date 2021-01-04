import numpy as np
import matplotlib.pyplot as plt




#############   parameters    ############
N=16
am=0.5
d=4
##########################################


########## definition of gamma matrices (weyl represention, euclidean metric) ##############

gamma_1=[ [  0.0,   0.0,    0.0, 1.0j],
          [  0.0,   0.0,   1.0j,  0.0],
          [  0.0, -1.0j,    0.0,  0.0],
          [-1.0j,   0.0,    0.0,  0.0] ]


gamma_2=[ [  0.0,   0.0,    0.0, 1.0],
          [  0.0,   0.0,   -1.0,  0.0],
          [  0.0, -1.0,    0.0,  0.0],
          [ 1.0,   0.0,    0.0,  0.0] ]

gamma_3= [ [  0.0,   0.0,    1.0j, 0.0],
          [  0.0,   0.0,   0.0,  -1.0j],
          [  -1.0j, 0.0,    0.0,  0.0],
          [ 0.0,   1.0j,    0.0,  0.0] ]


gamma_4= [ [ 0.0,   0.0,    1.0, 0.0],
          [  0.0,   0.0,   0.0,  1.0],
          [  1.0,   0.0,    0.0,  0.0],
          [  0.0,   1.0,    0.0,  0.0] ]

I4 = np.eye(4) #identity matrix of dim 4 by 4

#######################################################

px=np.zeros(N)

for i in range(N):
    px[i]=np.pi*i/N

py=px
pz=px
p4=px

l=N*N*N

aE=[]
ap=[]
aE_continuum=[]
eig_real=[]
eig_imag=[]


def Wilson_op_eig(p1, p2, p3, p4):
     term1 = np.add(        np.multiply( (1j)*np.sin(p1), gamma_1 )  ,  np.multiply(2*(np.sin(p1/2.0))*(np.sin(p1/2.0)), I4 )        )
     term2 = np.add(        np.multiply( (1j)*np.sin(p2), gamma_2 )  ,  np.multiply(2*(np.sin(p2/2.0))*(np.sin(p2/2.0)), I4 )        )
     term3 = np.add(        np.multiply( (1j)*np.sin(p3), gamma_3 )  , np.multiply(2*(np.sin(p3/2.0))*(np.sin(p3/2.0)), I4 )         )
     term4 = np.add(        np.multiply( (1j)*np.sin(p4), gamma_4 )  ,  np.multiply(2*(np.sin(p4/2.0))*(np.sin(p4/2.0)), I4 )        )
     D_wilson = np.add(term1, term2)
     D_wilson = np.add(D_wilson, term3)
     D_wilson = np.add(D_wilson, term4)
    
     eig_val, eig_vec= np.linalg.eig(D_wilson)
     return eig_val




for i  in range(N):
    for j in range(i, N):
        for k in range(j, N):
            #magnitude of momentum vector
            ap_ijk=np.sqrt( px[i]*px[i]  +  py[j]*py[j] +  pz[k]*pz[k]     )
            #sum of square of sine of momentum
            p_bar_sq = (np.sin(px[i]))**2 + (np.sin(py[j]))**2 + (np.sin(pz[k]))**2
            
            #sum of cosines of momenta
            sum_cos_ap = np.cos(px[i]) + np.cos(py[j])  +np.cos(pz[k]) 
            denom = (d+ am  - sum_cos_ap)
            num = 1 + p_bar_sq + denom**2

            cosh_aE_ijk=num/(2*denom)
            if (cosh_aE_ijk<0):
                print(cosh_aE_ijk)
            aE_ijk = np.arccosh(cosh_aE_ijk)

            aE=np.append(aE, [aE_ijk])
            ap=np.append(ap, [ap_ijk])

            aE_cont_ijk = np.sqrt( ap_ijk**2 + am**2)
            aE_continuum=np.append(aE_continuum, [  aE_cont_ijk   ])
   

           # eig_p = Wilson_op_eig(px[i], py[j], pz[k], aE_ijk)
           # eig_real = np.append(eig_real, np.real(eig_p))
           # eig_imag = np.append(eig_imag, np.imag(eig_p))
            









for i  in range(N):
    for j in range(i, N):
        for k in range(j, N):
            for l in range(k, N):
                eig_p = Wilson_op_eig(px[i], py[j], pz[k], p4[l])
                eig_real = np.append(eig_real, np.real(eig_p))
                eig_imag = np.append(eig_imag, np.imag(eig_p))
            
                
                              




file_name1 = 'wilson_dispersion_N='+str(N)+'_am='+str(am)+".dat"
np.savetxt( file_name1, np.transpose([ap, aE]))
print("The dispersion data of wilson fermions is saved in "+file_name1)


file_name2 = 'continuous_dispersion_am='+str(am)+'.dat'
np.savetxt( file_name2, np.transpose([ap, aE_continuum]))
print("The dispersion data of fermions without discretisation is saved in "+file_name2)


file_name3 = 'wilson_spectrum_N='+str(N)+'_am='+str(am)+".dat"
np.savetxt( file_name3, np.transpose([eig_real, eig_imag]))
print("The real and imaginary parts of the eigen values (spectrum data) of wilson operator is saved in  "+file_name3)




title_text1="L/a="+str(N)+",  am="+str(am)

plt.title(title_text1)
plt.plot(ap, aE, '.', color='black', label='naive discretisation')
plt.plot(ap, aE_continuum, '.', color='blue', label='continuum', alpha=0.47)
plt.legend()
plt.xlabel('ap')
plt.ylabel('aE')
plt.ylim(0, 2.5)
image_file='wilson_r=1_am='+str(am)+"_N="+str(N)+".png"
plt.savefig(image_file, format='png')

plt.figure()
title_text2="Eigen values of Wilson Operator, L/a="+str(N)+",  am="+str(am)
plt.title(title_text2)
plt.plot(eig_real, eig_imag, '.', color='black')
plt.xlabel('Real(Eigen value of Wilson Operator)')
plt.ylabel('Real(Eigen value of Wilson Operator)')
image_file='WilsonSpectrum(r=1,am='+str(am)+",N="+str(N)+").png"
plt.savefig(image_file, format='png')
plt.show()

