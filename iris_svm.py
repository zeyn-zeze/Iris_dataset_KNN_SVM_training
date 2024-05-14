#ilgili kütüphanelerin yüklenmesi
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import metrics

iris = load_iris() #iris veri setinin yüklemesi
x,y = iris.data,iris.target
k_size = range(1,11) #k aralığı
c = 1.0 #C parametresi Margin genişliğini belirlemektedir.
accuracy= np.zeros(len(k_size)) #başarı oranlarının tutulduğu dizi oluşturulur

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.35,random_state=109) #%35 u test %65 i eğitim


for i in k_size:
  #Svm classifier 
  clf = SVC(kernel='linear', C=c ) #Veriler doğrusal şekilde ayrılacağı için kernel lineer alınmıştır. 
  clf.fit(X_train,Y_train) # sınıfın model üzerinde eğitilmesi
  y_pred = clf.predict(X_test) #Model test verisi üzerinde tahmin yapar
  accuracy[i-1] = metrics.accuracy_score(Y_test, y_pred) # Doğruluk skoru hesaplanması
  print(f"{i}.iteration C value: {c}, Accuracy Score: {accuracy[i-1]}")
  c /= 2 # C değerini azaltma
  

best_accuracy  = 0 #en iyi başarı oranının tutulacağı değer
best_c_value = 0 #başarı oranının en iyi olduğu yerdeki C değerinin tutulacağı değer

best_accuracy = accuracy.max()
best_c_value = accuracy.argmax()+1
print(f"Best accuracy : {best_accuracy} , Best C value  : {best_c_value}")


# iterasyona göre skor değişkliğinin gösterildiği grafik
plt.plot(range(1, 10), accuracy, marker='o', linestyle='-')
plt.title('S Değerlerine Göre Doğruluk Skorları')
plt.xlabel('S Değeri')
plt.ylabel('Doğruluk')
plt.xticks(range(1, 10))
plt.grid(True)
plt.show()



  


