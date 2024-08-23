# ATC-neural-network

# Author

Jan Holáň

# Project description

The goal of this project is prediction of quantity and type of package, where the items will be packed in. The package type means palet or carton in general, but both of them has their own subtypes resulting in different sizes.

# Data description

jak postupovat pri ziskani dat , jejich trenovani a vyuziti natrenovane site.

priprava datasetu. CSV nebo json. 
1. z excel import csv ( autor Vandlik HonzaNeuronkaKD.csv - data o produktech,, HOnzaNeuronkaObal.csv - typy obalu) 
   - dalsi csv honzaneuronkzasilky, prijde zpakovany mailem.

      csv soubory importovat jako tabulky do SQL.
     na data v SQL pustit script.sql
2. vstupy (ATC_dataloader.py)- cisla boxu , rozmeny x,y,z
   
vstupy z 1 , spojil do celku train_in.csv  , jak - 

