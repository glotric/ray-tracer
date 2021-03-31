# Ray-Tracer
Projekt pri predmetu računalništvo (FMF, Univerza v Ljubljani).

Program prebere kompozicijo krogel it datotele podatki.txt in iz njih nariše realistično sliko. Kodo sem spisal sam s svojimi objekti, metodo delovanja pa sem povzel po [tutorial](https://medium.com/swlh/ray-tracing-from-scratch-in-python-41670e6a96f9). Dodal sem še ostrost predmetov na sliki glede na njihovo oddaljenost.

## Uporaba
Za uporabo je treba najprej pravilno oblikovati datoteko podatki.txt, potem pa zagnati program tracer.py. Zaslon je pravokotnik, ki se nahaja s središčem v (0,0,0) in seže po x-osi \[-1,1\] in po y-osi v skladu z razmerjem. Kamera naj bo na eni strani zaslona (pozitivna z koordinata), krogle pa na drugi.

#### Oblikoanje datoteke podatki.txt
Pod zaslon in kamera najprej napiši širino in višino zaslona v pixlih (celi števili ločeni s presledkom). V naslednji vrstici pa x, y in z koordinato kamere kot camera=(x,y,z).
Pod objekti pa našteješ vse krogle v kompoziciji, vsako v svojo vrstico oblike:
krogla R=d center=(x,y,z) RGB(r,g,b) obojnost=o

## Viri
[tutorial](https://medium.com/swlh/ray-tracing-from-scratch-in-python-41670e6a96f9)