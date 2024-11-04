from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
#Veri seti inceleme
iris = load_iris()

X = iris.data #feature
y = iris.target # target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Dt modeli olustur ve train et
tree_clf = DecisionTreeClassifier(criterion="gini", max_depth= 5, random_state=42) #criterion = "entropy"
tree_clf.fit(X_train, y_train)

#DT Evalation test
y_pred=tree_clf.predict(X_test)
accuracy =accuracy_score(y_test,y_pred)

print("İris veri seti ile  eigitilen Dt modeli dogurulugu", accuracy)

conf_matrix= confusion_matrix(y_test,y_pred)
print("Conf_ matrix" )
print(conf_matrix)

plt.figure(figsize=(15,10))
plot_tree(tree_clf,filled=True,feature_names= iris.feature_names,class_names=iris.target_names)#class_name= list(iris.target_names) list seklinde olması gerek ama ben hata almadim. 

feature_importances = tree_clf.feature_importances_ # En onemli node'um
feature_names = iris.feature_names

feature_importances_sorted=sorted(zip(feature_importances,feature_names), reverse=True) 
for importance, feature_name in feature_importances_sorted:
    print(f"{feature_name}: {importance}")
#https://chatgpt.com/share/67295834-2690-800d-a18d-f0ff5f6b51bb
