## Fraud Detection of Badan Jaminan Kesehatan Masyarakat Hackathon Dataset Using Convolutional-Neural-Network

### 1. BUSINESS UNDERSTANDING

##### Business Objective: 

Rumah sakit merupakan salah satu instansi yang bergerak sebagai pelayanan kesehatan bagi masyarakat. Dalam melaksanakan proses bisnisnya, peran BPJS cukup besar dalam mempengaruhi kualitas pelayanan bagi masyarakat. Namun dengan semakin banyak penggunaan BPJS Kesehatan, tidak jarang terjadi beberapa kecurangan (fraud) yang ditujukan untuk menguntungkan pihak tertentu. Pelaku yang terlibat bisa jadi adalah peserta BPJS Kesehatan, fasilitator kesehatan atau pembeli layanan kesehatan, penyedia obat dan alat kesehatan, dan pemangku kepentingan lainnya. Penanganan terkait masalah tersebut menjadi concern yang perlu untuk diatasi yang bertujuan untuk dapat mencegah dan mendeteksi berbagai indikasi potensi kecurangan sedini dan sesedikit mungkin. Sehingga dengan demikian biaya pelayanan kesehatan dapat dimanfaatkan semaksimal mungkin dalam memenuhi kepentingan dan pelayanan yang maksimal bagi masyarakat, serta untuk tetap menjaga sustainability BPJS Kesehatan.

##### Goal

Melakukan prediksi potensi terjadinya penyimpangan (fraud) pada klaim pelayanan Rumah Sakit.

##### Plan 

Melakukan tahapan data mining sesuai dengan metodologi data science CRISP-DM

![crisp-dm](https://user-images.githubusercontent.com/81342084/212624551-e9fd160f-d09f-4877-ae46-4548bc41bc84.png)

### 2. DATA UNDERSTANDING

Memuat informasi BPJS Kesehatan yang merupakan data publik mengenai aturan penamaan dan kesehatan 
secara umum. Data yang digunakan berukuran 10611501 dimana terdiri dari 200217 observasi dan 53 variabel dan memiliki proporsi kelas label pada data seimbang. 

1. Collecting Data : Collecting data merupakan proses pengumpulan, pengukuran serta analisis data yang digunakan dalam penelitian.

2. Describe data : Potensi terjadinya fraud pada klaim pelayanan Rumah Sakit maka set data yang digunakan dengan menggunakan algoritma Convolutional Neural Network (CNN) adalah fraud_detection_train dataset. Dataset ini terdiri dari 53 variable dengan total 200217 observasi
Karakteristik dataset:

![understanding](https://user-images.githubusercontent.com/81342084/212624729-a2bc18ac-a500-4ab4-bfb4-3265d6fd1612.jpg)

Visualisasi heatmap untuk melihat korelasi setiap fitur:

![heatmap](https://user-images.githubusercontent.com/81342084/212625089-8dc54b9c-ffb3-4e0d-a23b-d3dac3df2980.jpg)

3. Validation Data
Pada sub bagian ini berisi tahapan evaluasi, kelengkapan data dan kualitas data yang digunakan dalam mengerjakan proyek.
Terjadinya missing value maupun noise pada data diakibatkan karena terjadinya kesalahan maupun error pada
saat melakukan penginputan data.

### 3. DATA PREPARATION
Pada tahapan data preparation berikut akan dijabarkan proses menyiapkan data, pemilahan variabel yang akan dianalisis, serta pembersihan data
1. Data Selection :
Data selection atau feature selection digunakan untuk memilih beberapa feature untuk membangun model klasifikasi. Proses seleksi dilakukan dengan melakukan penggabungan terhadap feature yang terkait menjadi satu selanjutnya memilih feature
yang akan digunakan sebagai input feature

![data_selection](https://user-images.githubusercontent.com/81342084/212626640-d6c0724f-cc52-4325-8627-219e0b40678b.jpg)

2. Cleaning Data:
Data Cleaning merupakan proses persiapan data dengan cara menghapus atau memodifikasi data yang salah, tidak akurat, tidak terformat maupun duplikat. Data yang rusak tentunya akan berpengaruh pada kinerja pada sistem.

![data_cleaning](https://user-images.githubusercontent.com/81342084/212626668-e3cc77bb-8c10-49d4-8e0a-fac989b51503.jpg)

3. Construct Data:
Mengkonstruksi data merupakan bagian dari Data transformasi yang terdiri dari representasi fitur, menentukan korelasi dan mengintegrasikan data. representasi fitur digunakan untuk mengurangi kompleksitas, meningkatkan akurasi dan memilih fitur optimal.

![data_transform1](https://user-images.githubusercontent.com/81342084/212626718-2113dd1e-a845-451f-8baf-65c45653ec96.jpg)
![data_transform2](https://user-images.githubusercontent.com/81342084/212626727-4b484f67-36a8-46b2-9b6c-db098bbbf5b8.jpg)

4. Labelling Data:
Pada kasus Fraud Detection (Binary Classification) data dibagi menjadi data training dan data validation yang berbeda.
![labelling_data](https://user-images.githubusercontent.com/81342084/212626756-01b8fba7-a03c-46ea-9c5e-c9688b7a577a.jpg)

5. Data Integration:
Pada tahap mengintegrasi data dilakukan concatenation. Concatenation dapat dianggap sebagai sebuah pendekatan untuk menambahkan baris atau kolom ke data. Pendekatan ini dimungkinkan jika data terbagi menjadi beberapa bagian atau jika dilakukan perhitungan yang ingin ditambahkan ke set data yang sudah tersedia.

![data_integration](https://user-images.githubusercontent.com/81342084/212626783-69a2ef22-0432-4070-8711-05bc8d4efdef.jpg)

### 4. MODELLING
Pada bagian ini dijelaskan mengenai pemilihan teknik modelling, menghasilkan test design, membangun model
atau membuat pemodelan, dan menilai model yang telah dibangun. Model yang digunakan adalah Binary Classification using Convolutional Neural Network Algorithm

1. Building Test Scenario
Teknik pemodelan yang dilakukan pada penelitian melibatkan penerapan cnn dalam melakukan prediksi jumlah kasus dan unit cost pada sebuah daerah akibat penambahan Rumah Sakit dari 200217 observasi dan 53 variable. Adapun feature yang digunakan pada dataframe terdiri atas kdkc, dati2, typeppk, jkpst, umur, jnspelsep, los, cmg, severitylevel diagprimer untuk input feature serta label yang menjadi target feature.

![build_test](https://user-images.githubusercontent.com/81342084/212626797-21081144-9c9b-45f9-b965-a9b9691cb6f2.jpg)

![build_test_scenario](https://user-images.githubusercontent.com/81342084/212626808-0e2fd9d4-6d00-4d83-be02-a5e091ed58cb.jpg)

2. Build Model
Mendefenisikan model Convolutional Neural Network.

![model_cnn(1)](https://user-images.githubusercontent.com/81342084/212626969-21fcbfb4-ef6d-4be8-8f15-8e511c854770.jpg)

Berikut kode untuk melakukan compile model dan fit model cnn

![compile_modelcnn](https://user-images.githubusercontent.com/81342084/212626985-5dc744e4-0d75-436d-a4a4-ca2b11ab9cda.jpg)

Hasil akhir compile model dan fit model

![fit_model](https://user-images.githubusercontent.com/81342084/212626999-c5938fdc-9cab-4e66-98d0-40cbd2cd2f91.jpg)

### 5. Evaluation
Pada bagian ini dilakukan tahap Evaluation (Evaluasi) dengan tujuan untuk memprediksi seberapa baik model akhir akan
bekerja nantinya sehingga diketahui apakah model tersebut layak digunakan atau tidak dan untuk membantu menemukan model yang paling mewakili pelatihan data

Berikut ditampilkan hasil evaluasi terhadap model yang dikembangkan:

![evaluate](https://user-images.githubusercontent.com/81342084/212627155-8be43ee8-8ae8-4652-8f8c-46675de2407c.jpg)
