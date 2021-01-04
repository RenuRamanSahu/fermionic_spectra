import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as npoly

###### parameters  #######

N=32
am=0.5
d=4

###########################

px=np.zeros(N)

for i in range(N):
    px[i]=np.pi*i/N

py=px
pz=px
aE = []
ap = []
ap_continuum=[]
aE_continuum=[]

for i in range(N):
    for j in range(i, N):
        for k in range(j, N):
            ap_ijk = np.sqrt(   px[i]**2    +  py[j]**2 +   pz[k]**2  )

            #summ of sines 
            sum_sin = np.sin(px[i]) + np.sin(py[j]) + np.sin(pz[k])
            sum_cos = np.cos(px[i]) + np.cos(py[j]) + np.cos(pz[k])
 
            c1=0.*(1+0j)
            #coeff of first term
            for p in [px[i], py[j], pz[k]]:
                c1 = c1 + (np.sin(p)+ np.cos(p)-1.0)**2

            c2 = -2.0*(2.0+sum_sin)*(1+0j)
            c3 = 4.0j
            c4 = sum_sin*(3-sum_cos)*(1+0j)
            c5 = (2.0j)*(3-sum_cos)*(1+0j)
            c6 = (1+0j)*am**2

            b1=c1+c4+c6
            b2=c2
            b3=c3
            b4=c5
             

            #coefficients of the fourth order polynomial
            a4=b1-b2          #coefficient of t^4
            a3=b3-b4          #coefficient of t^3
            a2=-2.0*b1+b2     #coefficient of t^2
            a1=b4             #coefficient of t^1
            a0=b1             #coefficient of t^0

            a=(a0, a1, a2, a3, a4)
            roots = npoly.polyroots(a)
            
            aE_ijk = 2*np.arctanh(roots)
            aE_real_ijk = min(np.abs(np.real(aE_ijk)))
            
            aE = np.append(aE, [aE_real_ijk])
            ap = np.append(ap, [ap_ijk]) 


            ap_continuum=np.append(ap_continuum, [ap_ijk])
            aE_cont_ijk = np.sqrt( ap_ijk**2 + am**2)
            aE_continuum=np.append(aE_continuum, [  aE_cont_ijk   ]) 






title_text="Dispersion relation of BC fermions "+"  L/a="+str(N)+",  am="+str(am)

plt.title(title_text)
plt.plot(ap, aE, '.', color='black', label='BC discretisation')
plt.plot(ap_continuum, aE_continuum, '.', color='blue', label='continuum')
plt.legend()
plt.xlabel('ap')
plt.ylabel('aE')
plt.ylim(0, 2.5)
image_file='BC(r=1,am='+str(am)+",N="+str(N)+").png"

filename1='BC_r=1_am='+str(am)+"_N="+str(N)+".dat"
np.savetxt( filename1, np.transpose([ap, aE]))

filename2 = "am="+str(am)+"_continuous_dispersion.dat"
np.savetxt(filename2, np.transpose([ap_continuum, aE_continuum]))

plt.savefig(image_file, format='png')
plt.show()         






