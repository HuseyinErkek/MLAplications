#sklearn: ML icin kullanilan en populer kütüphanelerden birisi.

from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt


#(1) Veri Seti İncelemesi
cancer = load_breast_cancer()
df = pd.DataFrame(data = cancer.data, columns=cancer.feature_names)
df["target"] = cancer.target # Dataframe'e target sutunu ekledik.

#(2) Modelin Seçilmesi(KNN)
#(3) Modelin Train edilmesi
X = cancer.data #feature
y = cancer.target #target
# train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.3, random_state=42)
#Olceklendirme Normalizaston ve Standardizasyon
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
accuracy_value = []
k_values = []
for k in range(1,21):
    knn = KNeighborsClassifier(n_neighbors=k) #Model olusturma
    knn.fit(X_train, y_train) #Fit fonksiyonu samplesleri kullanarak(feature,target) knn algoriltamasi gelistirir.
    #(4) Sonuclaarin degerlendirilmesi
    y_pred= knn.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    accuracy_value.append(accuracy)
    k_values.append(k)
    #print("Dogruluk degeri ",accuracy)    
    #conf_matrix = confusion_matrix(y_test,y_pred)
    #print("Confusion Matrix ", conf_matrix)
#(5) Hiperparametre ayarlanmasi
"""
KNN: Hiperparametre : K
    K: 1,2,3 ... N

"""
plt.figure()
plt.plot(k_values, accuracy_value)
plt.title("K degerine gore dogruluk")
plt.xlabel("K degeri")
plt.ylabel("Dogruluk")
plt.xticks(k_values)
plt.grid(True)
