import numpy as np
import matplotlib.pyplot as plt


###### parameters  #######

N=16
am=0.5
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

l=N*N*N

aE=[]
ap=[]
aE_continuum=[]
ap_continuum=[]
eig_real=[]
eig_imag=[]


def KW_op_eig(p1, p2, p3, p4):
     term1 = np.add(        np.multiply( (1j)*np.sin(p1), gamma_1 )  ,  np.multiply((1j)*2*(np.sin(p1/2.0))*(np.sin(p1/2.0)), gamma_4 )  )
     term2 = np.add(        np.multiply( (1j)*np.sin(p2), gamma_2 )  ,  np.multiply((1j)*2*(np.sin(p2/2.0))*(np.sin(p2/2.0)), gamma_4 )  )
     term3 = np.add(        np.multiply( (1j)*np.sin(p3), gamma_3 )  , np.multiply((1j)*2*(np.sin(p3/2.0))*(np.sin(p3/2.0)),  gamma_4 )  )
     term4 = np.add(        np.multiply( (1j)*np.sin(p4), gamma_4 )  ,  np.multiply(0.0, I4 )        )
     D_kw = np.add(term1, term2)
     D_kw = np.add(D_kw, term3)
     D_kw = np.add(D_kw, term4)
    
     eig_val, eig_vec= np.linalg.eig(D_kw)
     return eig_val


for i  in range(N):
    for j in range(i, N):
        for k in range(j, N):
            #magnitude of momentum vector
            ap_ijk=np.sqrt( px[i]*px[i]  +  py[j]*py[j] +  pz[k]*pz[k]     )
            #sum of square of sine of momentum
            sum_sin_sq = (np.sin(px[i]))**2 + (np.sin(py[j]))**2 + (np.sin(pz[k]))**2
            
            #sum of cosines of momenta
            sum_cos_ap = np.cos(px[i]) + np.cos(py[j])  +np.cos(pz[k]) 
          
            pos_sinh_aE = (1.0j)*(3.0 - sum_cos_ap) + np.sqrt( sum_sin_sq  + (am)**2   )
            neg_sinh_aE = (1.0j)*(3.0 - sum_cos_ap) - np.sqrt( sum_sin_sq  + (am)**2   )


            pos_aE_ijk = np.arcsinh(pos_sinh_aE)
            neg_aE_ijk = np.arcsinh(neg_sinh_aE)
             



            aE=np.append(aE, [pos_aE_ijk])
            ap=np.append(ap, [ap_ijk])

            aE=np.append(aE, [neg_aE_ijk])
            ap=np.append(ap, [ap_ijk])


            ap_continuum=np.append(ap_continuum, [ap_ijk])
            aE_cont_ijk = np.sqrt( ap_ijk**2 + am**2)
            aE_continuum=np.append(aE_continuum, [  aE_cont_ijk   ])







for i  in range(N):
    for j in range(i, N):
        for k in range(j, N):
            for l in range(k, N):
                eig_p = KW_op_eig(px[i], py[j], pz[k], p4[l])
                eig_real = np.append(eig_real, np.real(eig_p))
                eig_imag = np.append(eig_imag, np.imag(eig_p))
            
                

eig_imag = np.sort(eig_imag)
index = np.arange(1, len(eig_imag)+1)





plt.figure()
title_text2="Eigen values of KW Operator, L/a="+str(N)+",  am="+str(am)
plt.title(title_text2)
plt.plot(index, eig_imag, '.', color='black')
plt.xlabel('index')
plt.ylabel('Im(Eigen value of KW Operator)')
image_file='Im_KW_eig(r=1,am='+str(am)+",N="+str(N)+").png"


filename3='KW_Im_eig_vs_index_r=1_am='+str(am)+"_N="+str(N)+".dat"
np.savetxt( filename3, np.transpose([index, eig_imag]))


plt.savefig(image_file, format='png')
plt.show()
