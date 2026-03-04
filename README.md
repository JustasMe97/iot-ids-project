# IoT tinklo atakų aptikimas naudojant mašininį mokymą

## Projekto aprašymas

Šio projekto tikslas – sukurti ir įvertinti mašininio mokymosi modelius, skirtus IoT tinklo srauto atakų aptikimui.  
Modeliai apmokomi naudojant **UNSW-NB15** duomenų rinkinį ir vertinami pagal klasifikacijos kokybę bei skaičiavimo sąnaudas.

Įgyvendinti modeliai:
- Random Forest
- MLP (Multi-Layer Perceptron)

---

## Naudojamas duomenų rinkinys

Naudojamas **UNSW-NB15** duomenų rinkinys.  
UNSW-NB15 duomenų rinkiniai turi būti atsisiųsti atskirai ir įdėti į `data/` katalogą.  
Nuoroda: https://research.unsw.edu.au/projects/unsw-nb15-dataset

Naudoti failai:

- `UNSW_NB15_training-set(in).csv`
- `UNSW_NB15_testing-set(in).csv`

Atliekama dvejetainė klasifikacija:
- `0` – normalus srautas  
- `1` – ataka  

Arba
- `multiclass` kelių klasių klasikifacija

---

## Naudojamos technologijos

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib

---

## Projekto struktūra
```bash
iot-ids-project/
│
├── data/                   # Duomenų rinkiniai
├── src/                    
│ ├── data_loader.py        # Duomenų įkėlimas
│ ├── preprocessing.py      # Duomenų paruošimas
│ ├── models.py             # Modelių apibrėžimas
│ ├── evaluation.py         # Vertinimo funkcijos
│ └── visualization.py      # Grafikai
│
├── main.py       # Pagrindinė vykdymo programa
├── README.md
└── requirements.txt
```

---

## Diegimas

Sukurk virtualią aplinką:

```bash
python -m venv venv
```
Aktyvavimas (Windows):

```bash
venv\Scripts\activate
```
Aktyvavimas (Linux/macOS):
```bash
source venv/bin/activate
```

Įdiek bibliotekas:
```bash
pip install pandas numpy scikit-learn matplotlib
```
Paleidimas
```bash
python main.py
```


Programa:
 - Įkelia duomenis
 - Paruošia duomenis
 - Apmoko modelius
 - Apskaičiuoja metrikas
 - Nubraižo ROC grafikus
 - Atspausdina modelių palyginimo lentelę

Du režimai: 
- binary (dvejetainė klasifikacija, 0-normalus srautas, 1-ataka)
- Multi-class klasifikacija pagal atakų tipus


