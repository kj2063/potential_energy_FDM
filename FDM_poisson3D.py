import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse as sp
import scipy.sparse.linalg
import scipy.linalg as la

N = 400

x_max = 40*3.08e19 #m
x_min = 0.1
y_max = 40*3.08e19
y_min = 0.1
z_max = 40*3.08e19 
z_min = 0.1

xi = np.linspace(x_min,x_max,N+2)
yj = np.linspace(y_min,y_max,N+2)
zk = np.linspace(z_min,z_max,N+2)

#if del_x = del_y = del_z = del_l

del_l = xi[1] - xi[0]

G = 6.67384e-11 #m^3 kg^-1 s^-2
M = 8.5e8*2e30 #kg

### make A matrix

A_d = (-6)*sp.eye(N) + sp.eye(N,k=1) + sp.eye(N,k=-1)
A_sd = sp.eye(N)

A_3d = sp.kron(sp.eye(N),A_d) + sp.kron(sp.eye(N,k=1),A_sd) + sp.kron(np.eye(N,k=-1),A_sd)
A_3sd = sp.eye(N*N)

A = sp.kron(sp.eye(N),A_3d) + sp.kron(sp.eye(N,k=1),A_3sd) + sp.kron(sp.eye(N,k=-1),A_3sd)

###make b matrix

#factor of miyamoto nagai potential model
a= 0
b= 0

def pontential_model(x,y,z):
    return (-G*M)/np.sqrt(x**2+ y**2 + (a + np.sqrt(z**2 + b**2))**2)

def density(x,y,z):
    return (M*(b**2)*(a*(x**2+ y**2) + (a + 3*np.sqrt(z**2 + b**2))*(a + np.sqrt(z**2 + b**2))**2)) /\
        (4*np.pi*(x**2+ y**2+(a + np.sqrt(z**2 + b**2))**2)**(5/2)*np.sqrt(z**2 + b**2)**(3))

B = np.zeros((N,N,N))

for k in range(N):
    for j in range(N):
        for i in range(N):
            B[k,j,i] += del_l**2 * 4 * np.pi * G * density(xi[i+1],yj[j+1],zk[k+1])

#add boundary condition in b matrix
for j in range(N):
    for i in range(N):
        B[0,j,i] += pontential_model(xi[i+1],yj[j+1],z_min)
        B[-1,j,i] += pontential_model(xi[i+1],yj[j+1],z_max)

for k in range(N):
    for j in range(N):
        B[k,j,0] += pontential_model(x_min,yj[j+1],zk[k+1])
        B[k,j,-1] += pontential_model(x_max,yj[j+1],zk[k+1])

for k in range(N):
    for i in range(N):
        B[k,0,i] += pontential_model(xi[i+1],y_min,zk[k+1])
        B[k,-1,i] += pontential_model(xi[i+1],y_max,zk[k+1])

B = B.reshape(N**3)
v = scipy.sparse.linalg.spsolve(A,B)
u = v.reshape(N,N,N)

U_bottom = np.zeros((1,N+2,N+2))
U_top = np.zeros((1,N+2,N+2))
U_left = np.zeros((N,N,1))
U_right = np.zeros((N,N,1))
U_front = np.zeros((N,1,N+2))
U_back = np.zeros((N,1,N+2))

for j in range(N+2):
    for i in range(N+2):
        U_bottom[0,j,i] += pontential_model(xi[i],yj[j],z_min)

for j in range(N+2):
    for i in range(N+2):
        U_top[0,j,i] += pontential_model(xi[i],yj[j],z_max)

for k in range(N):
    for j in range(N):
        U_left[k,j,0] += pontential_model(x_min,yj[j+1],zk[k+1])

for k in range(N):
    for j in range(N):
        U_right[k,j,0] += pontential_model(x_max,yj[j+1],zk[k+1])

for k in range(N):
    for i in range(N+2):
        U_front[k,0,i] += pontential_model(xi[i],y_min,zk[k+1])

for k in range(N):
    for i in range(N+2):
        U_back[k,0,i] += pontential_model(xi[i],y_max,zk[k+1])

U_1 = np.zeros((N,N+2,N+2))

for i in range(N):
    U_1[i] = np.vstack([U_front[i],np.hstack([U_left[i],u[i],U_right[i]]),U_back[i]])

U = np.vstack([U_bottom,U_1,U_top])
print(U.shape)

model_potential = np.zeros((N+2,N+2,N+2))

for k in range(N+2):
    for j in range(N+2):
        for i in range(N+2):
            model_potential[k,j,i] += pontential_model(xi[i],yj[j],zk[k])

U_rs = U.reshape((N+2)**3)
mp_rs = model_potential.reshape((N+2)**3)

x = range((N+2)**3)

fig, ax = plt.subplots()
ax.plot(x,np.abs(U_rs-mp_rs)/np.abs(mp_rs),".")
plt.show()
# cmap = mpl.cm.get_cmap("RdBu_r")

# fig = plt.figure(figsize =(19,5))
# ax = fig.add_subplot(111, projection='3d')
# counter = range(N+2)
# x,y,z = np.meshgrid(counter,counter,counter)
# ax.scatter(x,y,z, c=U.flat, cmap = cmap)
# # c = ax.pcolor(X,Y,U,vmin = -1e9, vmax=0, cmap=cmap)
# # cb = plt.colorbar(c, ax=ax, shrink=0.75)
# # cb.set_label("$u(R,Z)$",fontsize= 18)
# plt.show()