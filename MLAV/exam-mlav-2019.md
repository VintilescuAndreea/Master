YOLO algorithm is an algorithm based on regression, instead of selecting the interesting part of an Image, it predicts classes and bounding boxes for the whole image in one run of the Algorithm.

1. Trebuie construit un sistem machine learning care sa inchida automat o usa; o camera filmeaza intrarea si usa nu trebuie inchisa daca e cineva in ea. Descrieti componentele generale ale componentei de machine learning:

a) regresie vs clasificare

În principiu, clasificarea se referă la prezicerea unei etichete (ce apartine unei clase/valori discrete), iar regresia se referă la prezicerea unei cantități (o valoare continua dintr-un interval).

Pentru problema de detectie persoane si inchidere automata a usii daca o persoana se afla in incinta usii, va trebui sa folosim un algoritm bazat pe regresie (precum YOLO) pentru a identifica persoana si pozitia acesteia față de usa.

S-ar putea folosi si un sistem bazat pe clasificare, in care clasificam daca in cadrul curent avem sau nu o persoana, dar acest tip de sistem ar fi foarte rigid si ar trebui antrenat doar pe spatiile (sau spatii/locatii asemanatoare) cu locatia unde se va folosia camera de supraveghere.


b) date: antrenare/validare; se poate folosi un alt set de date si o procedura de transfer learning?

Datele de antrenare pot fi alcatuite din imagini ce contin persoane. Spre exemplu ne putem folosi de baze de date precum Caltech Pedestrian Dataset pentru a antrena un sistem YOLO care sa identifice persoane, iar atunci cand persoanele sunt identificate intr-o anumita zona intr-un cadru (o vecinatate manuala aleasa unde se afla zona usii), sistemul sa stie daca trebuie sa inchida usa sau nu.

Putem face transfer learning plecand de la modele YOLO antrenate pe baze de date mari precum ImageNet (care contin si o clasa de persoane).


c) Sistem de Machine Learning (tip, functie de cost, motivatie)

Algoritmul YOLO folosește rețele neuronale convoluționale (CNN) pentru a detecta obiecte în timp real. YOLO prezice mai multe casete de delimitare per celulă de grilă. Pentru a calcula pierderea pentru adevăratul pozitiv, vrem doar ca unul dintre ei să fie responsabil pentru obiect. În acest scop, îl selectăm pe cel cu cel mai mare IoU (intersecție peste unire) cu adevărul de bază.

d) Performanta asteptata si metoda de masura?

Pe setul de date ImageNet, YOLO asigură o precizie în top-1 de 76,5% și o precizie în top-5 de 93,3%, deci ne asteptam in general ca pe baza noastra de date sa avem o acuratete de detectie a persoanelor din usa de peste 75%.



2. Explicati in ce consta Selectia secventiala directa a trasaturilor (Features Selection).

Selectia secventiala directa si selectia secventiala inversa presupun ca eu selectez cel mai bine trasaturile/features care functioneaza cel mai bine cu clasificatorul pe care il construiesc.

Selectia secventiala directa:
- incepe cu un set vid la care urmeaza sa adaug in el
- incerc toate variantele/combinatiile cu 1 trasatura pentru clasificator (ex: daca am un set de 128 trasaturi atunci am 128 combinatii de incercat cu clasificatorul) si aleg trasatura cea mai buna (cea mai buna dpv performanta, unde metrica depinde de clasificator, exemplu acuratete/eroare patratica medie)
- dupa ce am ales 1 trasatura si o adaug in set, trebuie sa vad care va fi urmatoarea trasatura care va fi buna alaturi de (in combinatie cu) trasaturile alese existente deja in set (incerc iar toate combinatiile din trasaturile ramase)
- ma opresc cu adaugarea trasaturilor in set in momentul in care incepe sa scada acuratea/performanta indicata de metrica folosita (sau ma opresc cand ating un numar maxim de trasaturi ce le pot folosi datorita unui sistem cu resurse finite)

Din pct de vedere al programarii, selectarea celei mai bune trasaturi fara sa imi pese ce avem mai departe se numeste greedy (problema tehnicii greedy este ca este probabil ca rezultatul sa nu fie optim, fiind posibil ca alte combinatii de trasaturi care au fost omise de algoritm sa fi fost mai optime).

Alt dezavantaj este ca algoritmul se opreste in a adauga trasaturi in momentul cand scade acuratetea, dar este posibil ca acuratetea sa fi inceput sa creasca din nou dupa un anumit interval.


Selectia secventiala inversa:
- se incepe cu un set ce contine toate trasaturile (ex toate cele 128 valori ale trasaturilor)
- incep sa testez toate variantele/combinatiile de trasaturi dand afara din set cate 1 trasatura (sperand ca eliminand acea trasatura irelevanta, imi va creste performanta clasificatorului)
- conditia de stop este sa elimin trasaturi pana cand nu mai obtin crestere de performanta


Selectia trasaturilor random: 
- inseamna tehnica Greedy
- ex am disponibile 128 de trasaturi, unde pot incepe cu un set de 64 trasaturi selectate, si incep sa adaug sau sa elimin trasaturi in set in mod aleator
- adaug o trasatura random in set, daca imi creste performanta o pastrez, daca nu creste voi scoate trasatura din set si adaug alta.
- conditia de stop poate fi data de un anumit prag pe care ni-l dorim sa-l atingem ca performanta sau ne putem opri dupa un numar fix de incercari (de exemplu dupa 1000 de substitutii) - altfel ar trebui sa tin stocate toate combinatiile unde s-ar putea sa nu termin niciodata.

