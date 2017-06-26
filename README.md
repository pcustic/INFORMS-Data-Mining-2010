# INFORMS-Data-Mining-2010
Projekt u sklopu kolegija Strojno učenje na PMF-u pod nazivom Kratkotrajna kretanja u cijenama dionica.

## Opis datoteka i skript:
- TrainingData.csv - skup podataka za učenje modela.
- TestData.csv - skup podataka za testiranje modela.
- Training_VAR74.csv - samo Variable74 podaci iz skupa podataka za učenje.
- Test_VAR74.csv - samo Variable74 podaci iz skupa podataka za testiranje.
- LogReg.py - skripta koja stvara lagged podatke na ciljnoj varijabli sa koracima 1-12 i pokreće logističku regresiju na cijelom skupu podataka.
- LogReg_SVM_74_v1.py - skripta koja stvara lagged podatke na ciljnoj varijabli sa koracima 1-12 i pokreće logističku regresiju i SVM samo koristeci podatke od Variable74.
- LogReg_SVM_74_v2.py - skripta koja stvara lagged podatke na svim podacima Variable74 i ciljnoj varijabli sa koracima 1-12 i pokreće logističku regresiju i SVM.
- LogReg_SVM_74_v3.py - skripta koja stvara lagged podatke na svim podacima Variable74 sa koracima 1-12 i pokreće logističku regresiju i SVM.

Sve skripte osim LogReg.csv koriste skupove Training_VAR74.csv i Test_VAR74.csv.
