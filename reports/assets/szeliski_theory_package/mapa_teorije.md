# Mapa teorije iz Szeliski knjige za ovaj seminarski

Ovo je spisak delova koji su stvarno relevantni za trenutni `main.tex`. Nema potrebe da ubacuješ cela poglavlja knjige.

| Tema u radu | Szeliski | Štampana strana | PDF strana | Šta uzeti |
|---|---|---:|---:|---|
| Opšti pregled motion estimation | Ch. 8 uvod | 382-383 | 404-405 | Kratak uvod + Slika 8.1 opcionalno |
| Translaciono poravnanje i SSD | Sec. 8.1 | 384-386 | 406-408 | Eq. 8.1, brightness constancy, rezidual |
| Coarse-to-fine piramida | Sec. 8.1.1 | 387-388 | 409-410 | Eq. 8.15-8.16, objašnjenje velikih pomeraja |
| Lucas-Kanade linearizacija | Sec. 8.1.3 | 392-394 | 414-416 | Slika 8.2, Eq. 8.33-8.42, posebno Eq. 8.38 |
| Aperture problem | Sec. 8.1.3 | 395-397 | 417-419 | Slika 8.3, opis loše uslovljenih regiona |
| Learned motion models | Sec. 8.2.2 | 403-404 | 425-426 | Obavezno: Slika 8.6 i Eq. 8.66 - direktno je walking primer |
| Gusti optical flow | Sec. 8.4 | 409-411 | 431-433 | Eq. 8.69, Horn-Schunck data term Eq. 8.70, ograničenja brightness constancy |
| Regularizacija | Sec. 3.7.1 | 174-176 | 196-198 | Eq. 3.98 `E=Ed+lambda Es`, poenta smoothness/data kompromisa |
| Furijeova analiza | Sec. 3.4 | 133-136 | 155-158 | Slika 3.24, DFT Eq. 3.54, Parseval |
| PCA kao projekcija | Sec. 14.2.1 | 671-673 | 693-695 | Eq. 14.9-14.15, posebno projekcija i whitening |
| SVD | App. A.1.1 | 736-739 | 758-761 | Eq. A.1, A.5, Slika A.1, veza SVD-PCA |
| Linear least squares | App. A.2 | 742-743 | 764-765 | Eq. A.28-A.30 |

## Slike u paketu

- `figure_8_1_motion_estimation.png` - opciona opšta slika motion estimation-a.
- `figure_8_2_incremental_flow.png` - korisna uz Lucas-Kanade/Taylor linearizaciju.
- `figure_8_3_aperture_problem.png` - korisna za ograničenja optical flow-a.
- `figure_8_6_walking_motion_basis.png` - najvažnija slika za ovaj rad; originalni primer je hodanje.
- `figure_A_1_svd_pca.png` - geometrijska interpretacija SVD/PCA.
- `figure_3_24_fourier.png` - korisna uz temporalnu/frekvencijsku analizu.

## Šta NE možeš da pokriješ samo Szeliskim

Za sledeće delove trenutnog rada koristi druge reference, ne pokušavaj da ih na silu pripišeš knjizi:

1. **Action Quality Assessment (AQA)** - PECoP / AQA radovi.
2. **R3D-18 i 3D konvolucije kao konkretna arhitektura** - Tran et al. 2018 + Torchvision dokumentacija.
3. **Kinetics-400 pretraining i transfer learning pipeline** - odgovarajući R3D/Torchvision izvori.
4. **Ridge regresija kao konkretan estimator** - poseban ML/statistički izvor; Szeliski daje least squares i opštu regularizaciju, ali ne tvoj tačan Ridge pipeline.
5. **MAE, R2, Spearman i pairwise ranking** - navedi statističke/ML izvore po potrebi.
6. **AQA kliničke/gait reference** - CARE-PD, PECoP i drugi gait assessment izvori.

## Najmanji skup koji bih stvarno stavio u rad

Ako ne želiš da teorijsko poglavlje bude predugačko, zadrži:

- SSD + brightness constancy;
- coarse-to-fine u jednom pasusu;
- Eq. `Ix u + Iy v + It = 0`;
- aperture problem + Slika 8.3;
- dense optical flow;
- SVD/PCA + jedna formula za kovarijansu i projekciju;
- Eq. `u(x)=sum a_k v_k(x)` + Slika 8.6;
- DFT u jednom kratkom pododeljku;
- whitening jer ga stvarno koristiš u R3D grani.

To je dovoljno da teorija bude direktno vezana za implementaciju, bez prepričavanja pola knjige.
