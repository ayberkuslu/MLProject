

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

MLProject/XGBClassifier/XGBoostClassifier.ipynb gives best solution.


--Furkan Beğendi 161101043
--Elif Başak Yıldırım 161101032
--Salih Doruk Şahin 161101028
--Ayberk Uslu 161101055
