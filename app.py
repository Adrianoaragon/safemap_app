from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import pickle, json, os

app = Flask(__name__, static_folder='.', static_url_path='')

BASE = os.path.dirname(__file__)

# Load model and encoders once at startup
with open(os.path.join(BASE,'model.pkl'),'rb') as f: rf_model = pickle.load(f)
with open(os.path.join(BASE,'encoders.pkl'),'rb') as f: le_dict = pickle.load(f)
with open(os.path.join(BASE,'feature_cols.pkl'),'rb') as f: feature_cols = pickle.load(f)
with open(os.path.join(BASE,'frontend_data.json'), encoding='utf-8') as f: frontend_data = json.load(f)

MESES_MAP = {'Enero':1,'Febrero':2,'Marzo':3,'Abril':4,'Mayo':5,'Junio':6,
             'Julio':7,'Agosto':8,'Septiembre':9,'Octubre':10,'Noviembre':11,'Diciembre':12}
DIAS_MAP  = {'lun.':0,'mar.':1,'mié.':2,'jue.':3,'vie.':4,'sáb.':5,'dom.':6}
FRANJA_MAP = {'MADRUGADA (00:00-05:59)':0,'MAÑANA (06:00-11:59)':1,
              'TARDE (12:00-17:59)':2,'NOCHE (18:00-23:59)':3}

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/data')
def api_data():
    return jsonify(frontend_data)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        body = request.get_json()
        mes     = MESES_MAP.get(body['mes'], 1)
        dia     = DIAS_MAP.get(body['dia'], 0)
        franja  = FRANJA_MAP.get(body['franja'], 2)
        arma    = body['arma']
        edad    = body['edad']
        genero  = body['genero']
        zona    = body['zona']
        sitio   = body['sitio']
        barrio  = body['barrio']

        datos = {
            'Mes_Num':                 mes,
            'Dia_Num':                 dia,
            'Es_Finde':                1 if dia >= 5 else 0,
            'Franja_Num':              franja,
            'Armas / Medios_enc':      int(le_dict['Armas / Medios'].transform([arma])[0]),
            'Agrupa Edad Persona_enc': int(le_dict['Agrupa Edad Persona'].transform([edad])[0]),
            'Género_enc':              int(le_dict['Género'].transform([genero])[0]),
            'Zona_enc':                int(le_dict['Zona'].transform([zona])[0]),
            'Clase de Sitio_enc':      int(le_dict['Clase de Sitio'].transform([sitio])[0]),
            'Barrio_enc':              int(le_dict['Barrio'].transform([barrio])[0]),
        }
        X_new = pd.DataFrame([datos])[feature_cols]
        nivel = rf_model.predict(X_new)[0]
        probs = rf_model.predict_proba(X_new)[0]
        prob_dict = {c: round(float(p),4) for c,p in zip(rf_model.classes_, probs)}

        return jsonify({'nivel': nivel, 'probabilidades': prob_dict, 'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 400

@app.route('/api/sitios')
def sitios():
    return jsonify(list(le_dict['Clase de Sitio'].classes_))

if __name__ == '__main__':
    app.run(debug=True, port=5050)
