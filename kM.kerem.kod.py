# csv dosyalarını okumak için
import pandas as pd

# 2 boyutlu grafik oluşturmak için
import matplotlib.pyplot as plt

# Konunumu ve dosyayı görmediği için burada ufak bir kod ekledim

import os
os.getcwd()
os.chdir('C:\\Users\\Kkara\\Desktop\\İş Analatiği')
# verilerimi yükledim
data=pd.read_csv('Dataset.csv')

# başta cluster sayısını veriyorum
n_clust=4

data=pd.DataFrame(data)
# fiyata göre yapcagım kümeleme için sadece fiyatları çektim ve df adında DataFrame yaptım.
df= data.loc[:,['distance_travelled(kms)','price']]
#işlemleri rahat yapabilmek için array tipine çevirdim
df= df.values
# KMeans sınıfını import ettim
from sklearn.cluster import KMeans

kM = KMeans(n_clusters=n_clust, init='k-means++',random_state=0)
#kümeleme işlemini yapacağım
kM.fit(df)


# Tahmin işlemi yapıyoruz.
predict = kM.predict(df)


#küme merkezleri
##[[2983667.64705883]
## [ 783937.83371472]
##[7248716.21621622]]
print(kM.cluster_centers_)






#grafik oluşturma kısmı

#deneme yaptım farklı kaynağa bakarak ama  merkezleri gösteremedim

 
# import matplotlib.pyplot as plt
# plt.scatter(df[predict==0,0],df[predict==0,1],s=50,color='red')
# plt.scatter(df[predict==1,0],df[predict==1,1],s=50,color='blue')
# plt.scatter(df[predict==2,0],df[predict==2,1],s=50,color='green')

# plt.show()










colors = ['b', 'g', 'r', 'c']
markers = ['o', 'v', 's', 'p']
plt.ylabel('price')


for i in range(4):
    plt.scatter(df[predict == i, 0], df[predict == i, 1], s = 100, c = colors[i], marker=markers[i], label = 'Cluster - {:d}'.format(i+1))


plt.scatter(kM.cluster_centers_[:, 0], kM.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')

plt.xlabel('kms')
plt.legend()
plt.show()



# 4 küme kısmı ile yapılması belki daha sağlıklı olabilir.















