<h1 style="text-align: center"> <b>Segmentacja zdjęć satelitarnych</b> </h1>

<div style="text-align: center"> <b>Autorzy</b> </div>
<div style="text-align: center"> Witold Nowogórski, Szymon Jurecki, Arkadiusz Paterak </div>

## Opis problemu

## Metody rozwiązania problemu

### Klasyfikacja binarna fragmentów zdjęć

- samodzielnie przygotowany zbiór danych, adnotowany ręcznie przy użyciu opracowanego programu
- klasy: tło (0), zabudowania (1)
- model oparty na pretrenowanym modelu ResNet-34

### Segmentacja semantyczna przy pomocy modelu U-Net

- zbiór danych: [LandCover.ai v1](https://landcover.ai.linuxpolska.com/#dataset); zdjęcia satelitarne z Polski wraz z adnotacjami w postaci masek
- klasy: inne (0), zabudowania (1), tereny leśne (2), woda (3), drogi (4)
- model oparty na architekturze U-Net, zaimplementowany samodzielnie w bibliotece PyTorch

## Napotkane trudności i ich rozwiązania

- zbyt duży rozmiar zdjęć ze zbioru LandCover.ai spowalniał uczenie modelu U-Net
    - rozwiązanie: zdjęcia są skalowane do rozmiaru 64x64
- dobranie odpowiedniej funkcji straty dla modelu U-Net działającego dla wielu klas i określenie właściwych wymiarów wyjściowej warstwy (w wielu przykładach dokonywano segmentacji dla dwóch klas)
    - rozwiązanie: zastosowanie funkcji straty `nn.CrossEntropyLoss` oraz wyjściowej warstwy o wymiarach [batch_size, n_classes, image_height, image_width] (dla przypadku binarnego wystarcza jeden kanał w miejsce n_classes); inferencja dokonywana poprzez użycie funkcji softmax, a następnie wybór klasy o najwyższym prawdopodobieństwie

## Kontrybucje autorów

Witold Nowogórski stworzył program do ręcznej adnotacji danych oraz był pomysłodawcą pierwszej metody. Szymon Jurecki ... Arkadiusz Paterak zaproponował użycie modelu U-Net, który zaimplementował, stworzył także prosty interfejs do zaprezentowania działania modelu. Wszyscy autorzy wspólnie przygotowali raport.