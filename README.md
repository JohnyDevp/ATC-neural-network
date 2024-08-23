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
     na data v SQL pustit script.sql  - vytvari select , ktery ulozime do csv a mame soubory train_in.csv
     z druhe casti vystupu scriptu ulozime soubor train_out.csv
     test_in a _out , musi byt rozdilne data od train_in _out. 80% train a 20% test
     
2. ATC_dataloader.py - necte soubory train_in i train_out
   transformuje csv soubory do vektoru s pevnym rozmerem 5000, kdyz chybi data doplni se nulama.
   5000- maximalni pocet udaju o produktech na jedne objednavce.(groupdelivery = objednavka)

   obdobne vytvoreny vektor - rozmer 18,  boxu ze souboru train_out
dalo by se ulozit do file , nebo DB ,ale zustava jen v pameti.
   
atc_model.py - obsahuje customizaci predtrenovane NN.
tento model urcuje jak uspesne se neuronka natrenuje. 

nb.ipynb - zakladni py program,ktery vola jednotlive moduly. 
pytorch.org  web stranka - inpirace






