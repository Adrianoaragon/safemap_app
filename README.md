# SafeMap Barranquilla — Instrucciones de ejecución

## Requisitos
- Python 3.9+
- pip
- flask==3.0.3
- pandas==2.2.2
- numpy==1.26.4
- scikit-learn==1.5.0
- openpyxl==3.1.2

## Instalación y arranque

```bash
# 1. Instalar dependencias
pip install -r requirements.txt
pip install flask pandas scikit-learn openpyxl

# 2. Correr la app
python app.py
```

## Acceso
Abrir en el navegador: http://localhost:5050

## Estructura
safemap_barranquilla/ <br>
├── app.py                               ← Backend Flask (modelo + API) <br>
├── encoders.pkl                         ← LabelEncoders por variable <br>
├── feature_cols.pkl                     ← Lista de features del modelo <br>
├── frontend_data.json                   ← Datos agregados para el mapa <br>
├── hurto_personas_barranquilla_2025_2026.xlsx  ← Dataset fuente <br>
├── index.html                           ← Frontend (mapa + formulario) <br>
├── model.pkl                            ← Modelo Random Forest entrenado <br>
└── requirements.txt <br>

## Endpoints API
- GET  /            → Sirve el frontend.
- GET  /api/data    → Devuelve categorías y datos del mapa.
- POST /api/predict → Recibe parámetros, devuelve predicción RF.
- GET  /api/sitios  → Lista de clases de sitio disponibles.
