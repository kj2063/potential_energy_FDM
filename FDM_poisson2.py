import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.sparse as sp
import scipy.sparse.linalg


N = 100 #

R_max = 40*3.08e19 #m
R_min = 0 #m
Z_max = 40*3.08e19 #m
Z_min = 0 #m

Ri = np.linspace(R_min,R_max,N+2)
Zj = np.linspace(Z_min,Z_max,N+2)

del_R = Ri[1]-Ri[0]
del_Z = Zj[1]-Zj[0]

G = 6.67384e-11 #m^3 kg^-1 s^-2
M = 8.5e8*2e30 #kg

# # miyamoto nagai coefficient
# a = 1 #
# b = 1 #

# def potential_model(R,Z):
#     return (-G*M)/np.sqrt(R**2 + (a + np.sqrt(Z**2 + b**2))**2)

# ### boundary condition
# u0_b = potential_model(Ri,0)
# u0_t = potential_model(Ri,Zj[-1])
# u0_l = potential_model(0,Zj)
# u0_r = potential_model(Ri[-1],Zj)

### make A matrix
A_1d = np.zeros((N,N))

for i in range(N):
    A_1d[i,i] += - (4 * Ri[i+1])

for i in range(N-1):
    A_1d[i,i+1] += Ri[i+1] + (del_R/2)

for i in range(N-1):
    A_1d[i+1,i] += Ri[i+2] - (del_R/2)

A_side = np.zeros((N,N))

for i in range(N):
    A_side[i,i] += Ri[i+1]

A = sp.kron(sp.eye(N),A_1d) + sp.kron(sp.eye(N,k=1),A_side) + sp.kron(sp.eye(N,k=-1),A_side)

### make b matrix
# miyamoto nagai coefficient
a_input = 0 #kpc
b_input = 0.001 #kpc

a = a_input * 3.08e19
b = b_input * 3.08e19

def potential_model(R,Z):
    return (-G*M)/np.sqrt(R**2 + (a + np.sqrt(Z**2 + b**2))**2)

def density(R,Z):
    return (M*(b**2)*(a*(R**2) + (a + 3*np.sqrt(Z**2 + b**2))*(a + np.sqrt(Z**2 + b**2))**2)) /\
        (4*np.pi*(R**2+(a + np.sqrt(Z**2 + b**2))**2)**(5/2)*np.sqrt(Z**2 + b**2)**(3))

B = np.zeros((N,N))

for i in range(N):
    for j in range(N):
        B[j,i] += (del_R**2) * Ri[i+1] * 4 * np.pi * G * density(Ri[i+1],Zj[j+1])

#boundary condition
for i in range(N):
    B[0,i] += -Ri[i+1] * potential_model(Ri[i+1],Z_min)
    B[-1,i] += -Ri[i+1] * potential_model(Ri[i+1],Z_max)
    B[i,0] += -(Ri[1] - del_R/2) * potential_model(R_min,Zj[i+1])
    B[i,-1] += -(Ri[N] + del_R/2)  * potential_model(R_max,Zj[i+1])

B = B.reshape(N**2)

v = scipy.sparse.linalg.spsolve(A,B)
u = v.reshape(N,N)

U = np.vstack([potential_model(Ri,Z_min),np.hstack([potential_model(R_min,Zj[1:N+1]).reshape(N,1),u,potential_model(R_max,Zj[1:N+1]).reshape(N,1)]),potential_model(Ri,Z_max)])

###plot
X,Y = np.meshgrid(Ri,Zj)
fig = plt.figure(figsize=(19,5))
cmap = mpl.cm.get_cmap("RdBu_r")

ax = fig.subplots(1,3)
c = ax[0].pcolor(X,Y,U,vmin = -1e9, vmax=0, cmap=cmap)#
ax[0].set_title("$Result$", fontsize = 18)
ax[0].set_xlabel("R", fontsize = 18)
ax[0].set_ylabel("Z", fontsize = 18)
cb = plt.colorbar(c, ax=ax[0], shrink=0.75)
cb.set_label("$u(R,Z)$",fontsize= 18)


###test
model = np.zeros((N+2,N+2))
for i in range(N+2):
    for j in range(N+2):
        model[j,i] += potential_model(Ri[i],Zj[j])

print("### U")
print(U)
print("### model")
print(model)
print(".")
print(np.abs(np.abs(U-model)/model))

c1 = ax[1].pcolor(X,Y,model,vmin = -1e9, vmax=0, cmap=cmap)
ax[1].set_title("$Model$", fontsize = 18)
ax[1].set_xlabel("R", fontsize = 18)
ax[1].set_ylabel("Z", fontsize = 18)
cb1 = plt.colorbar(c1, ax=ax[1], shrink=0.75)
cb1.set_label("$u(R,Z)$",fontsize= 18)

c2 = ax[2].pcolor(X,Y,np.abs(np.abs(U-model)/model),vmin = 0, vmax=1e-3, cmap=cmap)
ax[2].set_title("$Result - Model$", fontsize = 18)
ax[2].set_xlabel("R", fontsize = 18)
ax[2].set_ylabel("Z", fontsize = 18)
cb2 = plt.colorbar(c2, ax=ax[2], shrink=0.75)
cb2.set_label("$u_{res} - u_{model}$",fontsize= 18)
plt.show()



grvf_R = np.zeros((N,N))

for j in range(N):
    for i in range(N):
        grvf_R[j,i] +=  (U[j,i+2]-U[j,i])/(2*del_R)

grvf_Z = np.zeros((N,N))

for j in range(N):
    for i in range(N):
        grvf_Z[j,i] +=  (U[j+2,i]-U[j,i])/(2*del_R)

tot_grvf = np.sqrt(grvf_R**2+grvf_Z**2)


tot_grvf = tot_grvf[1,:]

# tot_grvf = tot_grvf[]

# alpha = np.zeros((N+1,1))
# alpha2 = np.zeros((1,N+2))

# tot_grvf = np.hstack((tot_grvf,alpha))
# tot_grvf = np.vstack((tot_grvf,alpha2))

# def mu(x):
#     return x/(1+x)

def r(R,Z):
    return np.sqrt(R**2+Z**2)

def abs_gravi(R,Z):
    PD_R = G*M*R*(R**2+(a+np.sqrt(Z**2+b**2)**2))**(-3/2)
    PD_Z = G*M*Z*(1+(a/np.sqrt(Z**2+b**2)))*(R**2+(a+np.sqrt(Z**2+b**2))**2)**(-3/2)
    return np.sqrt(PD_R**2+PD_Z**2)

N_grvf = np.zeros((N,N))
Ri_N = np.linspace(R_min+del_R,R_max-del_R,N)
Zj_N = np.linspace(R_min+del_R,R_max-del_R,N)

for j in range(N):
    for i in range(N):
        N_grvf[j,i] += abs_gravi(Ri_N[i],Zj_N[j])

N_grvf = N_grvf[1,:]

# mod_mgrav = np.zeros((N,N))

# for j in range(N):
#     for i in range(N):
#         mod_mgrav[j,i] += mond_grav(Ri_N[i],Zj_N[j])

# mod_mgrav = mod_mgrav[1,:]


# N_grvf = N_grvf.reshape(51*101)
# tot_grvf = tot_grvf.reshape(51*101)
# N_grvf = N_grvf.reshape((N+1)**2)

# tot_grvf = tot_grvf.reshape((N+1)**2)

x = np.linspace(1e-11,1e-9)
y = np.linspace(1e-11,1e-9)

fig, ax = plt.subplots(1)
ax.plot(Ri_N,np.log10(tot_grvf))
ax.plot(Ri_N,np.log10(N_grvf))
# ax.set_xlim(-13,-9)
# ax.set_ylim(-13,-9)
plt.show()
