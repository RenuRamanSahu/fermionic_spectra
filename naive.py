import numpy as np
import matplotlib.pyplot as plt

N=32
am=0.5

px=np.zeros(N)

for i in range(N):
    px[i]=np.pi*i/N

py=px
pz=px

l=N*N*N

aE=[]
ap=[]
aE_continuum=[]

for i  in range(N):
    for j in range(i, N):
        for k in range(j, N):
            ap_ijk=np.sqrt( px[i]*px[i]  +  py[j]*py[j] +  pz[k]*pz[k]     )
            p_bar_sq = (np.sin(px[i]))**2 + (np.sin(py[j]))**2 + (np.sin(pz[k]))**2
            aE_ijk=np.arcsinh(np.sqrt(  p_bar_sq + (am)*(am)    ))

            aE=np.append(aE, [aE_ijk])
            ap=np.append(ap, [ap_ijk])

            aE_cont_ijk = np.sqrt( ap_ijk**2 + am**2)
            aE_continuum=np.append(aE_continuum, [  aE_cont_ijk   ])





file_name1='naive_dispersion_N='+str(N)+"_am="+str(am)+".dat"
np.savetxt( file_name1, np.transpose([ap, aE]))
print("The dispersion data of naive fermions is saved in "+file_name1)



file_name2="continuous_dispersion_am="+str(am)+".dat"
np.savetxt( file_name2, np.transpose([ap, aE_continuum]))
print("The dispersion data of fermions without discretisation is saved in "+file_name2)




title_text="L/a="+str(N)+",  am="+str(am)
plt.title(title_text)
plt.plot(ap, aE, '.', color='black', label='naive discretisation')
plt.plot(ap, aE_continuum, '.', color='blue', label='continuum')
plt.legend()
plt.xlabel('ap')
plt.ylabel('aE')
plt.ylim(0, 2.5)
image_file='naive_m'+str(am)+"_N"+str(N)+".png"
plt.savefig(image_file, format='png')
plt.show()







