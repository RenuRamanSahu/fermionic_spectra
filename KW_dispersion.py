import numpy as np
import matplotlib.pyplot as plt


###### parameters  #######

N=32
am=0.0
d=4

###########################


############## Computing The Dispersion Relation ######################## 
px=np.zeros(N)

for i in range(N):
    px[i]=np.pi*i/N

py=px
pz=px

l=N*N*N

aE=[]
ap=[]
aE_continuum=[]
ap_continuum=[]

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



title_text="Dispersion relation of KW fermions "+"  L/a="+str(N)+",  am="+str(am)

plt.title(title_text)
plt.plot(ap, np.real(aE), '.', color='black', label='KW discretisation')
filename1='KW_r=1_am='+str(am)+"_N="+str(N)+".dat"
np.savetxt(filename1, np.transpose([ap, np.abs(np.real(aE))]))

plt.plot(ap_continuum, aE_continuum, '.', color='blue', label='continuum', alpha=0.47)
filename2 = "am="+str(am)+"_continuous.dat"
np.savetxt(filename2, np.transpose([ap_continuum, aE_continuum]))

plt.legend()
plt.xlabel('ap')
plt.ylabel('aE')
plt.ylim(0, 2.5)
image_file='KW(r=1,am='+str(am)+",N="+str(N)+").png"
plt.savefig(image_file, format='png')
plt.show()
