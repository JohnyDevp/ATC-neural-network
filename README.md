# ATC-neural-network

# Author

Jan Holáň

# Project description

The goal of this project is prediction of quantity and type of package, where the items will be packed in. The package type means palet or carton in general, but both of them has their own subtypes resulting in different sizes.

# Prostředí pro trénování

Na tomto linku se nachází možnosti placeného prostředí pro vývoj u Googlu. Platí se množství výpočetních jednotek, které se použijí během výpočtu.
Odhad na experimentální trénování této neuronové sítě je zhruba 15h ~ 300 jednotek.

https://colab.research.google.com/signup?utm_source=dialog&utm_medium=link&utm_campaign=settings_page

# Popis práce neuronky

POZN.: NN - neuronová síť

1. Nutné nasbírání dat, určení si zdrojů (kontrola práv na taková data - z neuronek lze zjistit co za data byla použita, s ohledem na to je třeba být si jist licensema)

    - rozlišují se přístupy učení supervised (s dozorem) a unsupervised (bez dozoru). Liší se v tom, zda ke vstupním datům potřebujeme i ukázková výstupní. V tomto projektu taková data potřeba jsou a v mnoha dalších případech taky. Tedy data rozdělujeme na **vstupní** a **výstupní**, kdy samozřejmě po naučení sítě už taková síť bude dostávat pouze data vstupní (výstup požadujeme aby odpovídal tomu, co se ta síť naučila)
    - data v tomto projektu jsou z I6 od Michala Vandlíka, získaná pomocí nějakých SQL dotazů (zeptat se M. Vandlika), jejich forma je v _input_vandlik/raw_input.xlsx_
        - v souborech _kd.csv, obal.csv a zasilky.csv_ se nachazejí jednotlivé listy původního souboru v _.csv_ formátu, který lze dále nahrát do SQL databáze (konkrétně využívám MySQL, ale jakákoli jiná by toto měla umět také)
        - z nahraných souborů je dále vytaženo to, co je pro NN v tomto projektu důležité, pomocí SQL skriptu viz soubor _scripts/script.sql_ (pro detaily a vysvětlení je soubor okomentován, ale zmíním, že vytvářím 4 soubory - train*{out,in}.csv pro trénovací část a test*{out,in}.csv pro validační část)
            - **trénovací data** (cca 80%) jsou použita jen a pouze pro trénování, nesmí se na nich model vyhodnocovat, protože jinak by mohlo dojít ke zkreslení údajů (sí´t se umí relativně dobře naučit vstupy, na kterých byla trénovaná)
            - **validační data** (cca 20%) jsou použita jen a pouze pro vyhodnocování sítě a zjištění jak moc dobře si síť poradí s daty, které ještě neviděla
            - pro tento projekt jsou soubory s těmito daty uloženy v adresáři _data/_

2. Příprava dat

    - profilrování dat (nalezení těch, jejichž hodnota je _třeba_ neúplná - může se jednat v tomto projektu o třeba řádek zboží, kde nejsou uvedeny všechny parametry). V závislosti na aplikaci sítě se ale můžou například chybějící nebo zavádějící hodnoty doplńovat/měnit - záleží velmi na okolnostech a zvážení, jestli takový zásah síť ovlivní nebo ne.
    - normalizace dat (pokud by například v číslech byly velké rozdíly, které by mohly mít negativní dopad na fungování sítě). Zde v tomto projektu takový přístup není použit

3. Postavení sítě
    - (nejnáročnější část, třeba prostudovat sítě pro danou problematiku)
    -

# Data description

jak postupovat pri ziskani dat , jejich trenovani a vyuziti natrenovane site.

priprava datasetu. CSV nebo json.

1. z excel import csv ( autor Vandlik HonzaNeuronkaKD.csv - data o produktech,, HOnzaNeuronkaObal.csv - typy obalu)

    - dalsi csv honzaneuronkzasilky, prijde zpakovany mailem.

        csv soubory importovat jako tabulky do SQL.
        na data v SQL pustit script.sql - vytvari select , ktery ulozime do csv a mame soubory train_in.csv
        z druhe casti vystupu scriptu ulozime soubor train_out.csv
        test_in a \_out , musi byt rozdilne data od train_in \_out. 80% train a 20% test

2. ATC_dataloader.py - necte soubory train_in i train_out
   transformuje csv soubory do vektoru s pevnym rozmerem 5000, kdyz chybi data doplni se nulama.
   5000- maximalni pocet udaju o produktech na jedne objednavce.(groupdelivery = objednavka)

    obdobne vytvoreny vektor - rozmer 18, boxu ze souboru train_out
    dalo by se ulozit do file , nebo DB ,ale zustava jen v pameti.

atc_model.py - obsahuje customizaci predtrenovane NN.
tento model urcuje jak uspesne se neuronka natrenuje.

nb.ipynb - zakladni py program,ktery vola jednotlive moduly.
pytorch.org web stranka - inpirace
