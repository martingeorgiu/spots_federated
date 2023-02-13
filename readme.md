# spots distributed

## Usage

You can run following `./bin/run-proper-flower.sh` to run a real local federation with 3 clients using FedAvg with 50 rounds.

## TODO

- weight decay
- momentum
- chytrejsi rozdeleni datasetu aby vahy byly stejne
- lepsi normalizace
- nejake vic kompenzace nerovnomerneho datasetu?
- mozne dalsi pokusy
  - oversampling a trochu undersampling

## versions

- v10 - prvni poradne funkcni reseni 8eaeb3128d4078c4edf1fcd22d3d423501673d7c
- v17 - zkousim weights k cross entropy - 926d75a73acc314d3cad931ff0feb46fce899b57
- v18 - zmensil jsem ty weights oproti jednicce, ale nahoby stagnujici vysledek - e9c03c0c82af9abf1125961ddd34949d4248190c
- v19 - lr 0.001 - 63098fa92c06ce004beea67c4ba724fa05af6ad8
- v26 - spravne learning rate;focal loss;pretrained weights;cisty dataset bez nasobeni;augmentace training; ea2e9b7dd6a01f71c2ef1b31f44703de266fc988
- v27 - cross entropy se spravnymy vahami - 3c03b514d5081670f7f98664165a6b4cee26eb74
- v28 - weights podle train datasetu
- v30 - fix seg faultu crossentropy s training vahami 30epoch d077db96ce8c1052be119c5d765c497301d8e244
- v31 - focal loss na 50 epoch alfa nula gamma 2 cde2b33dba51b2075d0b88fbbf9a2ee3deaf4971
- v32 - stejny jako v31 ale ulozil jsem rozlozeni train/val/test + train shuffle - 17afafaa45a820dd9938f95600034f0c870795a5
- buildeni slusnych modelu pres federaci 7efd91ec2244b2ce719feb919866271a52d0b836
- buildim ekvivalent simulaci federated-models/2023-02-11T01:00:29 sem federated-models-simulated/2023-02-12T12:42:35 af382d5d41f8203f53bb09a0026017846cd424b2
