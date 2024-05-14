# Iris Dataset Classification with K-Nearest Neighbors (KNN) and Support Vector Machine (SVM)

Bu proje, ünlü Iris veri setini kullanarak K-Nearest Neighbors (KNN) ve Support Vector Machine (SVM) algoritmalarını uygulamak için bir örnektir.

## K-Nearest Neighbors (KNN) Algoritması

KNN algoritması , sınıflandırma için kullanılan denetimli makine öğrenme algoritmasıdır. Komşulara bakarak tahmin yürütülür ve benzer olan sınıflar birbirlerine yakın olanlardır. KNN içinde tahmin edilecek değeri en yakın komşularından hangisinde yoğun olduğu bilgisine  dayanarak sınıfı tahmin etmeye çalışılır.
Sınıflandırmanın yapılabilmesi için çeşitli parametrelere ihtiyaç duyulur: uzaklık,komşuluk sayısı(k) ve ağırlıklandırma ölçütleri . 
Uzaklığı tahmin edilecek değerin diğer noktalara uzaklıkları hesaplanır.Bunun için Öklid,Minkowski,Manhattan vb gibi uzaklık hesaplamaları kullanılabilir.
Komşuluk sayısı(k) KNN classı için önemli bir parametredir. Bu parametreye dayalı olarak sınıflandırmalar yapılmaktadır. K=1 alındığında hedef değer en yakın komşuya atanır. K nin gittikçe hedef değere yaklaşması veya uzaklaşmasına bağlı olarak veri setindeki bilgiler dahilinde sınıf atamaları ve seçimleri yapılmaktadır. 


![image](https://github.com/zeyn-zeze/Iris_dataset_KNN_SVM_training/assets/116917341/204811b2-a4ae-46de-9653-a6e821104d36)


K parametresinin çok düşük seçildiği durumlarda model **overfit** olabilmektedir. Yani model eğitim setindeki durumları ezberlemiştir ve test edilen veri setinde bu durumları aramaktadır.Bu da test veri setinde kötü tahmin skorlarına sebep olabilir.k değerinin çok yüksek seçildiği durumda ise model **underfit** olabilmektedir. Model eğitim verilerine uymaz ve yeni verilen için genelleştirme yapılamaz. Bu sebeple hem eğitim hem de test veri setinde problemlerle karşılaşabilinmektedir. Bu sebeplerle k değerinin olabilecek en olası optimum değer seçilmesi gerekmektedir.

Diğer parametre olan ağırlık (weight) için uniform ve distance seçenekleri vardır. Ağırlığın uniform seçilmesi durumunda komşulukta bulunan tüm ağırlıklar eşit olacaktır.Distance seçilmesi durumunda ise komşulukta bulunan tüm komşuların ağırlığı dataya olan mesafe azaldıkça artacaktır. Bu hesaplama ise tüm komşuların ağırlığının (1 )/d  veya 1/d^2  şeklinde alınması ile yapılmaktadır. (d = hedef dataya mesafe)


### Algoritma İşleyişi
1. Öncelikle, veri seti eğitim ve test setleri olarak ayrılır.
2. Ardından, KNN algoritması için bir K değeri seçilir. Bu değer, yeni bir veri noktasının sınıflandırılmasında kullanılacak komşu sayısını belirler.
3. Eğitim seti üzerinde model eğitilir.
4. Test seti üzerinde modelin performansı değerlendirilir ve doğruluk skoru hesaplanır.
5. Farklı K değerleri için doğruluk skorları karşılaştırılarak en iyi K değeri bulunur.

KNN algoritmasının iris dataseti üzerinde eğitildiği ve test edildiğini gösteren kod parçasına [buradan](https://github.com/zeyn-zeze/Iris_dataset_KNN_SVM_training/blob/main/iris_knn.py) ulaşabilirsiniz

Kodda hespalanan doğruluk skorlarının gösterildiği grafik aşağıdaki gibidir

![image](https://github.com/zeyn-zeze/Iris_dataset_KNN_SVM_training/assets/116917341/f92ce5cb-d769-459c-976c-75b1ea93475a)

__Kullanım açısından KNN algoritması en yaklaşık değeri vermesine rağmen bellek kullanımı , veri seti ,maliyette artış ve birçok parametreye bağlı olarak etkilenmesi dezavantajlarına da sahiptir.__

## Support Vector Machine (SVM) Algoritması

SVM algoritması iki sınıf arasındaki en ayırt edici sınırın olmasını sağlayan hiper-düzlemin bulunması ile yürütülen sınıflandırma algoritmasıdır.
İki ya da daha fazla sınıflı veri kümelerinde , sınıfları ayırma işlemi SVM modellemesiyle yapılabilir. Ayrım içi belirlenmesi gereken karar sınırı ve en yakın veri noktası arasındaki mesafeye marjin denir. SVM iki sınıfı ayırabilmek için marjini en iyi olacak şekilde ayarlayarak bir hiper-düzlem oluşturur.Hiper-düzlem oluştururken karar sınırına en yakın ve destek olabilecek destek vektörleri olan veri noktaları kullanılır.

![image](https://github.com/zeyn-zeze/Iris_dataset_KNN_SVM_training/assets/116917341/9b2ada03-4a6c-4915-8082-94c6e5bad0d2)

SVC sınıflandırılması için kullanılan önemli parametreler şu şekildedir:
- C = Düzenleştirme parametresi.Bu parametre SVM modelinin karmaşıklığı kontrol etmesine ve overfitting problemlerini giderilmesine yardımcı olur.C değerinin artmasıyla SVM modelinin eğitilmesi de artacaktır. C değeri marjin genişliğinin etkilediği için C değerinin artması marjin değerinde küçülmeye sebep olacaktır.
- Kernel = Veri noktalarını ,düzlemsel olarak işlevsel hale getirmek için kullanılan matematiksel işlemlerdir.Verilerin ayrılma şekillerine göre farklı değerler almaktadır. Veri doğrusal olarak ayrılabiliyorsa __‘linear’__ değerini almalıdır. Doğrusal olarak ayrılmayan bir veri seti varsa __‘poly’__, __‘rbf’__, __‘sigmoid’__, __‘precomputed’__ değerleri kullanılır.
- Degree = Eğer baz alınan yöntem polinomsal yöntemse , bu parametre polinomun derecesini belirtir.
- Gamma = Eğer baz alınan yöntem RBF ise gamma RBF in genişliğini belirler ve karar sınırlarının esnekliğini ayarlar.

SVM modelleri yüksek eğitim süreleri sebebiyle büyük veri kümeleri için uygun değildir.Sınıflararası çakışma durumlarında da çalışma durumu iyi değildir. Fakat diğer modellemelere göre daha az bellek tüketimi ve hızlı tahmin görüsü sayesinde iyi çalışmalar gösterir.

### Algoritma İşleyişi
1. Öncelikle, veri seti eğitim ve test setleri olarak ayrılır.
2. SVM algoritması için C ve kernel gibi hiperparametreler belirlenir. C, marjın genişliğini kontrol ederken, kernel işlevi veri noktalarını yüksek boyutlu uzayda haritalamak için kullanılır.
3. Eğitim seti üzerinde model eğitilir.
4. Test seti üzerinde modelin performansı değerlendirilir ve doğruluk skoru hesaplanır.
5. Farklı hiperparametre değerleri için doğruluk skorları karşılaştırılarak en iyi hiperparametreler bulunur.

SVM modelinin iris dataseti üzerinde eğitim ve test sürecini gösteren kod parçasına [buradan](https://github.com/zeyn-zeze/Iris_dataset_KNN_SVM_training/blob/main/iris_svm.py) ulaşabilirsiniz

Kodda hesaplanan Doğruluk değerlerinin grafikte gösterimi aşağıdaki gibidir.
![image](https://github.com/zeyn-zeze/Iris_dataset_KNN_SVM_training/assets/116917341/9b4f5473-f609-420d-8c98-ebb657be77e8) 

(S:iterasyon değeri)

