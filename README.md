# Ray-Tracer
Projekt pri predmetu računalništvo (FMF, Univerza v Ljubljani).

Program prebere kompozicijo krogel it datotele podatki.json in iz njih nariše realistično sliko. Kodo sem spisal sam s svojimi objekti, metodo delovanja pa sem povzel po [tutorialu](https://medium.com/swlh/ray-tracing-from-scratch-in-python-41670e6a96f9).

## Uporaba
Za uporabo je treba najprej pravilno oblikovati datoteko podatki.json, potem pa zagnati program tracer.py. Zaslon je pravokotnik, ki se nahaja s središčem v (0, 0, 0) in seže po x-osi \[-1, 1\] in po y-osi v skladu z razmerjem višina/širina. Kamera naj bo na eni strani zaslona (pozitivna z koordinata), krogle pa na drugi.

#### Oblikoanje datoteke podatki.json
Json objekt naj ima 6 ključev: 
* "height", "width", in "depth" so integerji;
* "camera" je array dolžine 3 s koordinatami kamere;
* "light" je objekt s štirimi ključi, vsak od njih je array dolžine 3;
* "sphere" je poljubno dolg array objektov.

## Viri
[tutorial](https://medium.com/swlh/ray-tracing-from-scratch-in-python-41670e6a96f9)
