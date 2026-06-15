# SmartStock CG

Sistem za predikciju potraznje i optimizaciju zaliha.
Projekat iz predmeta Data Mining, FIST UDG.

## Opis

Hibridni ML sistem koji predvidja dnevnu prodaju po prodavnici,
racuna optimalnu kolicinu narudzbe (EOQ), safety stock,
detektuje anomalije i generise narudzbenice.

## Dataset

Rossmann Store Sales (Kaggle) - 1.017.000+ redova, 1.115 prodavnica.
Dataset nije ukljucen u repozitorijum zbog velicine,
treba ga skinuti sa Kaggle-a u `data/` folder.

## Stack

- Python 3.10
- pandas, numpy, scikit-learn
- XGBoost (forecasting)
- Prophet (baseline)
- Streamlit + matplotlib (web app)

## Pokretanje
## Rezultati

- XGBoost globalni model: SMAPE 8.24%, MAE 549 EUR
- Prophet baseline: SMAPE 11.16%, MAE 790 EUR
- Test period: zadnjih 42 dana

## Autor

OgnjenObradovic 21/026, FIST
