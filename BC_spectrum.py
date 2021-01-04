import numpy as np
import matplotlib.pyplot as plt


###### parameters  #######

N=8
am=0.0
d=4

###########################

px=np.zeros(N)

for i in range(N):
    px[i]=np.pi*i/N

py=px
pz=px
p4=px







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


Gamma = [[ 0. +0.j ,  0. +0.j ,  0.5+0.5j,  0.5+0.5j],
         [ 0. +0.j ,  0. +0.j , -0.5+0.5j,  0.5-0.5j],
         [ 0.5-0.5j, -0.5-0.5j,  0. +0.j ,  0. +0.j ],
         [ 0.5-0.5j,  0.5+0.5j,  0. +0.j ,  0. +0.j ]]




gamma1_dash = np.add(Gamma,  np.multiply(-1.0, gamma_1))
gamma2_dash = np.add(Gamma,  np.multiply(-1.0, gamma_2))
gamma3_dash = np.add(Gamma,  np.multiply(-1.0, gamma_3))
gamma4_dash = np.add(Gamma,  np.multiply(-1.0, gamma_4))


I4 = np.eye(4) #identity matrix of dim 4 by 4

############################################################################################
#print(np.dot(Gamma, Gamma))





         





def BC_fermion_eig(p1, p2, p3, p4, r):
     term1 = np.add(np.multiply( (1j)*np.sin(p1), gamma_1 )  ,  np.multiply((1j)*r*2*(np.sin(p1/2.0))*(np.sin(p1/2.0)), gamma4_dash ))
     term2 = np.add(np.multiply( (1j)*np.sin(p2), gamma_2 )  ,  np.multiply((1j)*r*2*(np.sin(p2/2.0))*(np.sin(p2/2.0)), gamma4_dash ))
     term3 = np.add(np.multiply( (1j)*np.sin(p3), gamma_3 )  , np.multiply((1j)*r*2*(np.sin(p3/2.0))*(np.sin(p3/2.0)),  gamma4_dash ))
     term4 = np.add(np.multiply( (1j)*np.sin(p4), gamma_4 )  ,  np.multiply((1j)*r*2*(np.sin(p4/2.0))*(np.sin(p4/2.0)),  gamma4_dash ))
     D_bc = np.add(term1, term2)
     D_bc = np.add(D_bc, term3)
     D_bc = np.add(D_bc, term4)
     D_bc = np.add(D_bc, np.multiply(am, I4))
    
     eig_val, eig_vec= np.linalg.eig(D_bc)
     return eig_val





eig_real = []
eig_imag = []

for i  in range(N):
    for j in range(i, N):
        for k in range(j, N):
            for l in range(k, N):
                eig_p = BC_fermion_eig(px[i], py[j], pz[k], p4[l], 1.0)
                eig_real = np.append(eig_real, np.real(eig_p))
                eig_imag = np.append(eig_imag, np.imag(eig_p))
            
                

eig_imag = np.sort(eig_imag)
index = np.arange(1, len(eig_imag)+1)





plt.figure()
title_text="Eigen values of BC fermion Operator, L/a="+str(N)+",  am="+str(am)
plt.title(title_text)
plt.plot(index, eig_imag, '.', color='black')
plt.xlabel('index')
plt.ylabel('Im(Eigen value of BC fermion Operator)')
image_file='Im_BC_eig(r=1,am='+str(am)+",N="+str(N)+").png"


filename='Im_BC_eig_r=1_am='+str(am)+"_N="+str(N)+".dat"
np.savetxt(filename, np.transpose([index, eig_imag]))

plt.savefig(image_file, format='png')
plt.show()

















