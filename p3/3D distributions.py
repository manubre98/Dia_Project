from user_classes import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

n = 10
bids = np.linspace(1.2, 1.2, n)
prices = np.linspace(5, 14, n)
mat = np.zeros(shape=(n, n))

for i in range(len(bids)):
    for j in range(len(prices)):
        mat[i, j] = obj_fun([classes[3]], bids[i], prices[j])

imax, jmax = np.unravel_index(np.argmax(mat), (n, n))
print(bids[imax], prices[jmax], np.max(mat))

plt.imshow(mat, cmap='hot')
plt.colorbar()
plt.show()

BIDS, PRICES = np.meshgrid(bids, prices)
OBJ = obj_fun(classes, BIDS, PRICES)

ax = plt.axes(projection='3d')
ax.plot_surface(BIDS, PRICES, OBJ, rstride=1, cstride=1, cmap='hot', edgecolor='none')
ax.set_title('surface')
plt.show()
