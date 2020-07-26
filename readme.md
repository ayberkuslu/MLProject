

--FEATURE ENGINEERING--
--Record tarihleri, date den geçtiği zamana göre inte çevrildi
-- String olan columnlar factorize edildi (int e dönüştü)
-- target column int'e donüştürüldü
-- birbirine çok yakın columnlar silindi
***********
-- Train seti 80'e 20 olarak ayrıldı.
-- Bu setler üzerinden denemeler gerçekleşti, optimal paramtreler bulundu
-- Daha sonra bu paramtrelerle tüm train seti eğitildi ve submission data predict edildi
-- predictionlar geri stringe çevrildi.


water-pump-3 ---> pump-predictions  0.8079 
xgboost_v2_probabilities ---> test_xgboost_d6  0.8079 