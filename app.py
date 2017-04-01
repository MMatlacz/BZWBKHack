# coding=utf-8
import json
from copy import deepcopy

import falcon
import numpy as np
import pandas as pd
from sklearn.externals import joblib

encoders = joblib.load('./label_encoders.bin')

app = falcon.API()

categorical = ['STAN_CYW', 'WYKSZTALCENIE', 'TYP_DOCHODU', 'KRAJ']

USER = {'DOCHOD': 0.0,
        'KLIENT_ID': 637732.0,
        'KRAJ': 0.0,
        'KWOTA_TRANS_K_DEB': 603.03999999999996,
        'KWOTA_TRANS_K_KRED': 1042.2858979913351,
        'KWOTA_TRANS_RACH_BIEZ': 6056.2799999999997,
        'KWOTA_y': 468.85999999999996,
        'LICZBA_K_DEB': 3.0,
        'LICZBA_OSOB_GOSP': 0.0,
        'LICZBA_PLAT_K_DEB': 7.0,
        'LICZBA_PROD_KRED': 0.0,
        'LICZBA_PROD_OSZCZ': 0.0,
        'LICZBA_RACH_BIEZ': 3.0,
        'LICZBA_RACH_K_KRED': 0.0,
        'LICZBA_TRANS_K_KRED': 10.726467113036628,
        'LICZBA_TRANS_RACH_BIEZ': 21.0,
        'LICZBA_WYPLAT_K_DEB_ATM': 2.0,
        'POSIADA_NIERUCHOMOSC': 0.0,
        'STAN_CYW': 3.0,
        'STAZ_KLIENT': 5569.0,
        'TYP_DOCHODU': 10.0,
        'WIEK': 40.0,
        'WYKSZTALCENIE': 4.0}

USER1 = {'DOCHOD': 0.0,
         'KLIENT_ID': 1235083.0,
         'KRAJ': 0.0,
         'KWOTA_TRANS_K_DEB': 0.0,
         'KWOTA_TRANS_K_KRED': 1042.2858979913351,
         'KWOTA_TRANS_RACH_BIEZ': 0.0,
         'KWOTA_y': 52.230000000000004,
         'LICZBA_K_DEB': 0.0,
         'LICZBA_OSOB_GOSP': 2.0,
         'LICZBA_PLAT_K_DEB': 0.0,
         'LICZBA_PROD_KRED': 0.0,
         'LICZBA_PROD_OSZCZ': 1.0,
         'LICZBA_RACH_BIEZ': 0.0,
         'LICZBA_RACH_K_KRED': 0.0,
         'LICZBA_TRANS_K_KRED': 10.726467113036628,
         'LICZBA_TRANS_RACH_BIEZ': 0.0,
         'LICZBA_WYPLAT_K_DEB_ATM': 0.0,
         'POSIADA_NIERUCHOMOSC': 2.0,
         'STAN_CYW': 0.0,
         'STAZ_KLIENT': 3821.0,
         'TYP_DOCHODU': 7.0,
         'WIEK': 34.0,
         'WYKSZTALCENIE': 5.0}

USERS = {
    USER['KLIENT_ID']: USER,
    USER1['KLIENT_ID']: USER1
}

cols = ['KLIENT_ID', 'STAN_CYW', 'WYKSZTALCENIE',
        'LICZBA_OSOB_GOSP',
        'DOCHOD', 'POSIADA_NIERUCHOMOSC', 'TYP_DOCHODU', 'WIEK']


class Mappings:
    def on_get(self, req, resp):
        resp.body = json.dumps({
            'STAN_CYWILNY': {
                'W': 'PANNA/KAWALER',
                'M': 'ZONATY/MEZATKA',
                'R': 'ROZWIEDZIONY/ROZWIEDZIONA',
                'U': 'WDOWA/WDOWIEC',
                'S': 'SEPARACJA'
            },
            'WYKSZTALCENIE': {
                'I': 'INNE',
                'L': 'LICENCJAT',
                'P': 'PODSTAWOWE',
                'S': 'SREDNIE',
                'W': 'WYZSZE',
                'Z': 'ZAWODOWE'
            },
            'TYP_DOCHODU': {
                0: 'INNE',
                1: 'UMOWA O PRACE',
                2: 'WLASNA DZIAŁALNOSC',
                3: 'EMERYTURA',
                4: 'RENTA',
                5: 'STYPENDIUM',
                6: 'RODZINA',
                7: 'NAJEM LUB DZIERZAWA',
                8: 'UMOWA O DZIELO/AGENCYJNA',
                9: 'UMOWA ZLECENIE'
            }
        })


class Login:
    last_id = None

    def on_get(self, req, resp):
        if req.params['user'] == 'user1':
            data = {k: v for k, v in USER.items() if k in cols}
            print(encoders.keys())
            data = {
                k: encoders[k].inverse_transform([int(v)])[
                    0] if k in encoders else v
                for
                k, v in data.items()}
        else:
            data = {k: v for k, v in USER1.items() if k in cols}
            print(encoders.keys())
            data = {
                k: encoders[k].inverse_transform([int(v)])[
                    0] if k in encoders else v
                for
                k, v in data.items()}
        Login.last_id = data['KLIENT_ID']
        resp.body = json.dumps(data)


class Country:
    def __init__(self):
        self.model = joblib.load('./kraje_clf.bin')
        self.ohencoder = joblib.load('./ohencoder.bin')

        self.ohencoder.handle_unknown = 'ignore'

        self.pca = joblib.load('./pca.bin')
        self.countries_mapping = ['AUT', 'CZE', 'DEU', 'DNK', 'IRL', 'ISL',
                                  'NLD', 'SWE']

    def process_data(self, result_json):
        local_user = USERS[Login.last_id]
        data_oryginal = deepcopy(local_user)
        for col in cols:
            if col in result_json:
                if col == 'TYP_DOCHODU':
                    data_oryginal[col] = \
                        encoders[col].transform([float(result_json[col])])[0]
                else:
                    data_oryginal[col] = \
                        encoders[col].transform([result_json[col]])[0]

        # resp.body = json.dumps(data_oryginal)

        df = pd.DataFrame.from_dict([data_oryginal])

        df[categorical] = df[categorical].astype(int)

        locations = [df.columns.get_loc(col) for col in categorical]

        self.ohencoder.categorical_features = locations

        df = self.ohencoder.transform(df)

        print(df.shape)

        df = self.pca.transform(df)

        return df

    def on_get(self, req, resp):
        result_json = req.params

        try:
            df = self.process_data(result_json)
        except:
            df = self.process_data({})

        prediction = self.model.predict_proba(df)

        prediction = np.array(prediction)[:, 0, 1]

        resp.body = json.dumps({
            'result': [c for b, c in zip(prediction, self.countries_mapping)
                       if b != 0]
        })


app.add_route('/login', Login())
app.add_route('/country', Country())
app.add_route('/mappings', Mappings())
