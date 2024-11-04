import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
X =np.sort(5 * np.random.rand(40,1),axis = 0)  #Uniform features 
y = np.sin(X).ravel()


# plt.scatter(X,y)
# add noise
y[::5] += 1 * (0.5 - np.random.rand(8))
#plt.scatter(X, y)
T = np.linspace(0, 5, 500)[:, np.newaxis] # 0 İLE 5 ARASINDA 500 ESİT ARALIKLI DEGER URETİR VE 500,1S SÜTÜN VEKTORUNE DONUSTURUR
for i, weight in enumerate(["uniform", "distance"]): # UNİFORM ESİT AGIRLIKLA DEGERLENDİRİR, DİSTANCE YAKİN KOMSULARA DAHA FAZLA UZAK KOMSULAR DAHA AZ AGİRLİK VERİR 
    knn = KNeighborsRegressor(n_neighbors=5, weights=weight)
    y_pred = knn.fit(X,y).predict(T)
    plt.subplot(2, 1, i + 1)
    plt.scatter(X, y, color = "green", label = "data")
    ,
    plt.plot(T, y_pred, color = "blue", label = "prediction")
    plt. axis("tight")
    plt. legend()
    plt.title("KNN Regressor weights = {}".format(weight))
plt.tight_layout()
plt.show()
#https://chatgpt.com/share/67294222-79d8-800d-91e3-1f88a188043c