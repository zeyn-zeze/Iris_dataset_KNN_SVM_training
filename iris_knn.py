#ilgili kütüphanelerin yüklenmesi 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#İris veri setinin yüklenmesi
iris = load_iris()

x = iris.data    #veri setinin özellikleri x e atanır
y = iris.target  #veri setinin hedef değişkenleri y e atanır



#Veri Setleri eğitim ve test seti olarak ikiye ayrılır.X_train ve Y_train değişkenleri modelin öğrenmesi için kullanılacak olan değişkenlerdir. X_test ve Y_test ise modelin performansını değerlendirmek için kullanılan parametrelerdir. 

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.35, random_state=0)

#Veri setinin %35 ı test seti olarak ayrılmıştır.Bu da test_size parametresi ile belirlenmiştir.
#random_state parametresi belirlendiğinde train_test_split fonksiyonu veri setini aynı şekilde böler

# En iyi k değerini bulan fonksiyon
def find_best_k(accuracy_table):

    best_k = accuracy_table.argmax()+1
    best_accuracy = accuracy_table.max()
    print(f"The best K value is {best_k} with {best_accuracy:.2f} accuracy(max)")
    


def evaluate_knn(K_size, X_train, X_test, Y_train, Y_test):

    accuracy_table = np.zeros(K_size)  # K_size boyutunda dizi oluşturulur

    for i in range(1, K_size+1):  # 1'den K_size+1'e kadar olan tüm k değerleri için döngü

        neighbor = KNeighborsClassifier(n_neighbors= i)  # KNeighborsClassifier nesnesi içindeki n_neighbors parametresi bu algoritmada kullanılacak olan komşu sayısını belirtir.

        neighbor.fit(X_train, Y_train) #Modelin eğitim verileri ile eğitilmesi

        y_pred = neighbor.predict(X_test) #Model test verisi üzerinde tahmin yapar

        accuracy_table[i-1] = accuracy_score(Y_test, y_pred) #her bir k değeri için hesaplanan doğruluk değeri diziye atılır

    return accuracy_table
    

# K-NN alogritması ile farklı k değerlerini değerlendirerek en iyi k değeri bulunmasını sağlar
accuracy_table = evaluate_knn(20, X_train, X_test, Y_train, Y_test) #değerlendirilecek k değeri 30 alınımıştır.

for k,accuracy in enumerate(accuracy_table,start=1): 
  print(f"Accuracy for K = {k} : {accuracy:.2f}")

find_best_k(accuracy_table) # max k değeri


# K değerlerini x ekseni, doğruluk skorlarını y ekseni olarak alarak grafiği çiz
plt.plot(range(1, 31), accuracy_table, marker='o', linestyle='-')
plt.title('K Değerlerine Göre Doğruluk Skorları')
plt.xlabel('K Değeri')
plt.ylabel('Doğruluk')
plt.xticks(range(1, 31))
plt.grid(True)
plt.show()

