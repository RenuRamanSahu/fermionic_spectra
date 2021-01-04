import numpy as np
import matplotlib.pyplot as plt


###### parameters  #######

N=8
am=0.0
d=4

###########################


















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
############## Computing The Dispersion Relation ######################## 
px=np.zeros(N)

for i in range(N):
    px[i]=np.pi*i/N

py=px
pz=px
p4=px

wilson_index=[]

l=N*N*N

aE=[]
ap=[]
aE_continuum=[]
ap_continuum=[]
eig_real=[]
eig_imag=[]



def KW_op_eig(p1, p2, p3, p4, r):
     term1 = np.add(        np.multiply( (1j)*np.sin(p1), gamma_1 )  ,  np.multiply((1j)*r*2*(np.sin(p1/2.0))*(np.sin(p1/2.0)), gamma_4 )  )
     term2 = np.add(        np.multiply( (1j)*np.sin(p2), gamma_2 )  ,  np.multiply((1j)*r*2*(np.sin(p2/2.0))*(np.sin(p2/2.0)), gamma_4 )  )
     term3 = np.add(        np.multiply( (1j)*np.sin(p3), gamma_3 )  , np.multiply((1j)*r*2*(np.sin(p3/2.0))*(np.sin(p3/2.0)),  gamma_4 )  )
     term4 = np.add(        np.multiply( (1j)*np.sin(p4), gamma_4 )  ,  np.multiply(0.0, I4 )        )
     D_kw = np.add(term1, term2)
     D_kw = np.add(D_kw, term3)
     D_kw = np.add(D_kw, term4)
    
     eig_val, eig_vec= np.linalg.eig(D_kw)
     return eig_val






for r in np.arange(0.0, 1.6, 0.1):
    for i  in range(N):
        for j in range(i, N):
            for k in range(j, N):
                for l in range(k, N):
                    eig_p = KW_op_eig(px[i], py[j], pz[k], p4[l], r)
                    eig_real = np.append(eig_real, np.real(eig_p))
                    eig_imag = np.append(eig_imag, np.imag(eig_p))
                    wilson_index = np.append(wilson_index, r*np.ones(4))
            
                



plt.figure()
title_text2="Eigen values of KW Operator vs r, L/a="+str(N)+",  am="+str(am)
plt.title(title_text2)
plt.plot(wilson_index, eig_imag, '.', color='black')
plt.xlabel('r')
plt.ylabel('Im(Eigen value of KW Operator)')
image_file='Im_KWeig_vs_r(am='+str(am)+",N="+str(N)+").png"


filename1='Im_KWeig_vs_r_am='+str(am)+"_N="+str(N)+".dat"
np.savetxt( filename1, np.transpose([wilson_index, eig_imag]))
plt.savefig(image_file, format='png')



plt.figure()
positive_eig_imag=np.abs(eig_imag)
title_text1="Eigen values of KW Operator vs r, L/a="+str(N)+",  am="+str(am)
plt.title(title_text1)
plt.plot(wilson_index, positive_eig_imag, '.', color='black')
plt.xlabel('r')
plt.yscale('log')
plt.ylabel('Im(Eigen value of KW Operator)')
image_file='log_Im_KWeig_vs_r(am='+str(am)+",N="+str(N)+").png"

filename2='positive_Im_KWeig_vs_r_am='+str(am)+"_N="+str(N)+".dat"
np.savetxt( filename2, np.transpose([wilson_index, positive_eig_imag]))

plt.savefig(image_file, format='png')

plt.show()
