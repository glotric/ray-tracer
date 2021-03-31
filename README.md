# ray-tracer
Projekt pri predmetu računalništvo.

Program prebere kompozicijo krogel it datotele podatki.txt in iz njih nariše realistično sliko.

[tutorial](https://medium.com/swlh/ray-tracing-from-scratch-in-python-41670e6a96f9)

## Oblikoanje datoteke podatki.txt
Pod zaslon in kamera najprej napiši širino in višino zaslona v pixlih (celi števili ločeni s presledkom). V naslednji vrstici pa x, y in z koordinato kamere kot camera=(x,y,z).
Pod objekti pa našteješ vse krogle v kompoziciji, vsako v svojo vrstico oblike:
krogla R=d center=(x,y,z) RGB(r,g,b) obojnost=o