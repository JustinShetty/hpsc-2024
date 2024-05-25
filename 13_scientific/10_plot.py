import matplotlib.pyplot as plt
import numpy as np

NX = 41
NY = 41

def main():
    x = np.linspace(0, 2, NX)
    y = np.linspace(0, 2, NY)
    X, Y = np.meshgrid(x, y)

    u = np.zeros((NY, NX))
    v = np.zeros((NY, NX))
    p = np.zeros((NY, NX))

    with open('u.dat', 'r') as f:
        udata = f.readlines()
    with open('v.dat', 'r') as f:
        vdata = f.readlines()
    with open('p.dat', 'r') as f:
        pdata = f.readlines()
    
    for n in range(len(udata)):
        plt.clf()
        u_flattened = udata[n].strip().split()
        u_flattened = [float(val) for val in u_flattened if val]
        v_flattened = vdata[n].strip().split()
        v_flattened = [float(val) for val in v_flattened if val]
        p_flattened = pdata[n].strip().split()
        p_flattened = [float(val) for val in p_flattened if val]

        for j in range(NY):
            for i in range(NX):
                u[j, i] = u_flattened[j * NX + i]
                v[j, i] = v_flattened[j * NX + i]
                p[j, i] = p_flattened[j * NX + i]

        plt.contourf(X, Y, p, alpha=0.5, cmap=plt.cm.coolwarm)
        plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
        plt.title(f'n = {n}')
        plt.pause(.01)
    plt.show()

if __name__ == '__main__':
    main()