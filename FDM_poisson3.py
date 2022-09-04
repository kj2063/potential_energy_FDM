import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.sparse as sp
import scipy.sparse.linalg


N = 100 #

R_max = 40*3.08e19 #
R_min = 0 #
Z_max = 40*3.08e19 #
Z_min = 0#

Ri = np.linspace(R_min,R_max,N+2)
Zj = np.linspace(Z_min,Z_max,N+2)

del_R = Ri[1]-Ri[0]
del_Z = Zj[1]-Zj[0]

G = 6.67384e-11 #m^3 kg^-1 s^-2
M = 8.5e8*2e30 #

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
    A_1d[i,i] += - del_R - (4 * Ri[i+1])

for i in range(N-1):
    A_1d[i,i+1] += Ri[i+1] +del_R

for i in range(N-1):
    A_1d[i+1,i] += Ri[i+2]

A_side = np.zeros((N,N))

for i in range(N):
    A_side[i,i] += Ri[i+1]

A = sp.kron(sp.eye(N),A_1d) + sp.kron(sp.eye(N,k=1),A_side) + sp.kron(sp.eye(N,k=-1),A_side)

### make b matrix
# miyamoto nagai coefficient
a = 1 #
b = 1 #

def potential_model(R,Z):
    return (-G*M)/np.sqrt(R**2 + (a + np.sqrt(Z**2 + b**2))**2)

def density(R,Z):
    return (M*(b**2)*(a*(R**2) + (a + 3*(Z**2 + b**2)**(1/2))*(a + (Z**2 + b**2)**(1/2))**2)) /\
        (4*np.pi*(R**2+(a + (Z**2 + b**2)**(1/2))**2)**(5/2)*(Z**2 + b**2)**(3/2))

B = np.zeros((N,N))

for i in range(N):
    for j in range(N):
        B[j,i] += (del_R**2) * Ri[i+1] * 4 * np.pi * G * density(Ri[i+1],Zj[j+1])

#boundary condition
for i in range(N):
    B[0,i] += -Ri[i+1] * potential_model(Ri[i+1],Z_min)
    B[-1,i] += -Ri[i+1] * potential_model(Ri[i+1],Z_max)
    B[i,0] += -Ri[1] * potential_model(R_min,Zj[i+1])
    B[i,-1] += -(Ri[N] + del_R) * potential_model(R_max,Zj[i+1])

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

print(U)
print(".")
print(model)
print(".")
print(U-model)

c1 = ax[1].pcolor(X,Y,model,vmin = -1e9, vmax=0, cmap=cmap)
ax[1].set_title("$Model$", fontsize = 18)
ax[1].set_xlabel("R", fontsize = 18)
ax[1].set_ylabel("Z", fontsize = 18)
cb1 = plt.colorbar(c1, ax=ax[1], shrink=0.75)
cb1.set_label("$u(R,Z)$",fontsize= 18)

c2 = ax[2].pcolor(X,Y,np.abs(np.abs(U-model)/model),vmin = 0, vmax=1e-4, cmap=cmap)
ax[2].set_title("$Result - Model$", fontsize = 18)
ax[2].set_xlabel("R", fontsize = 18)
ax[2].set_ylabel("Z", fontsize = 18)
cb2 = plt.colorbar(c2, ax=ax[2], shrink=0.75)
cb2.set_label("$u_{res} - u_{model}$",fontsize= 18)
plt.show()