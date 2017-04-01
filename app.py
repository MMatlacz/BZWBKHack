# coding=utf-8
import json
from copy import deepcopy
from datetime import datetime, timedelta

import falcon
import numpy as np
import pandas as pd
import requests
from falcon_cors import CORS
from sklearn.externals import joblib

from data_provider import DataProvider
from skyscanner_live_pricing import LivePricing
from weather import get_temp

cors = CORS(allow_origins_list=['*'])
public_cors = CORS(allow_all_origins=True)

encoders = joblib.load('./label_encoders.bin')

app = falcon.API(middleware=[cors.middleware])

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
         'POSIADA_NIERUCHOMOSC': 1.0,
         'STAN_CYW': 0.0,
         'STAZ_KLIENT': 3821.0,
         'TYP_DOCHODU': 7.0,
         'WIEK': 34.0,
         'WYKSZTALCENIE': 5.0}

USER2 = {'DOCHOD': 0.0,
         'KLIENT_ID': 1003788.0,
         'KRAJ': 0.0,
         'KWOTA_TRANS_K_DEB': 1002.5156079872838,
         'KWOTA_TRANS_K_KRED': 1042.2858979913351,
         'KWOTA_TRANS_RACH_BIEZ': 19747.093683204283,
         'KWOTA_y': 170.03000000000003,
         'LICZBA_K_DEB': 0.0,
         'LICZBA_OSOB_GOSP': 0.0,
         'LICZBA_PLAT_K_DEB': 17.653486985893107,
         'LICZBA_PROD_KRED': 0.0,
         'LICZBA_PROD_OSZCZ': 0.0,
         'LICZBA_RACH_BIEZ': 0.0,
         'LICZBA_RACH_K_KRED': 0.0,
         'LICZBA_TRANS_K_KRED': 10.726467113036628,
         'LICZBA_TRANS_RACH_BIEZ': 39.110601280948281,
         'LICZBA_WYPLAT_K_DEB_ATM': 3.090403337969402,
         'POSIADA_NIERUCHOMOSC': 2.0,
         'STAN_CYW': 3.0,
         'STAZ_KLIENT': 4789.0,
         'TYP_DOCHODU': 0.0,
         'WIEK': 63.0,
         'WYKSZTALCENIE': 4.0}

USERS = {
    'user': USER,
    'user1': USER1,
    'user2': USER2
}

cols = ['KLIENT_ID', 'STAN_CYW', 'WYKSZTALCENIE',
        'LICZBA_OSOB_GOSP',
        'DOCHOD', 'POSIADA_NIERUCHOMOSC', 'TYP_DOCHODU', 'WIEK']


class Mappings:
    cors = public_cors

    mapping = {
        'STAN_CYW': {
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
            'Z': 'ZAWODOWE',
            'U': 'NIEZNANE'
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
            9: 'UMOWA ZLECENIE',
            10: 'NIEZNANY'
        },
        'POSIADA_NIERUCHOMOSC': {
            0: "NIE POSIADA",
            1: "POSIADA",
            2: "BRAK DANYCH"
        }
    }

    @staticmethod
    def convert(category, value):
        try:
            return Mappings.mapping[category][value]
        except:
            print(category, value)

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
                9: 'UMOWA ZLECENIE',
                10: 'NIEZNANY'
            }
        })


class Login:
    last_id = None
    cors = public_cors

    def on_get(self, req, resp):
        if not req.params:
            raise falcon.HTTPBadRequest(
                description='Please add user parameter')
        user = USERS[req.params['user']]

        data = {k: v for k, v in user.items() if k in cols}
        data = {
            k: encoders[k].inverse_transform([int(v)])[
                0] if k in encoders else v
            for
            k, v in data.items()}

        print(data)

        random_user = requests.get('https://randomuser.me/api')
        random_user = random_user.json()['results'][0]
        data['name'] = '{} {}'.format(random_user['name']['first'],
                                      random_user['name']['last'])
        data['picture'] = random_user['picture']['medium']
        data = {k: (Mappings.convert(k, v) if k in Mappings.mapping else  v)
                for k, v in data.items()}

        Login.last_id = req.params['user']
        print(Login.last_id)
        resp.body = json.dumps(data)


class Country:
    cors = public_cors

    def __init__(self):
        self.model = joblib.load('./kraje_clf.bin')
        self.ohencoder = joblib.load('./ohencoder.bin')

        self.ohencoder.handle_unknown = 'ignore'

        self.pca = joblib.load('./pca.bin')
        self.countries_mapping = ['AUT', 'CZE', 'DEU', 'DNK', 'IRL', 'ISL',
                                  'NLD', 'SWE']

    def process_data(self, result_json):
        try:
            local_user = USERS[Login.last_id]
        except Exception:
            raise falcon.HTTPBadRequest(description='Please login first')
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

        df = self.pca.transform(df)

        return df

    def on_get(self, req, resp):
        result_json = req.params

        try:
            df = self.process_data(result_json)
        except Exception:
            df = self.process_data({})

        prediction = self.model.predict_proba(df)

        prediction = np.array(prediction)[:, 0, 1]

        resp.body = json.dumps({
            'result': [c for b, c in zip(prediction, self.countries_mapping)
                       if b != 0]
        })


class Flights:
    cors = public_cors

    def get_flight(self, inbound, cheapest):
        outbound = 'Warszawa Chopina'

        outbound_date = (datetime.today() + timedelta(days=1)).date()
        inbound_date = (datetime.today() + timedelta(days=7)).date()

        country_info = requests.get(
            'https://restcountries.eu/rest/v2/alpha/{}'.format(
                inbound)).json()

        city = country_info['capital']
        name = country_info['name']

        try:
            cheapest[name] = LivePricing(
                DataProvider.get_suggestions(outbound)[0]['code'].split(
                    '-')[0],
                DataProvider.get_suggestions(city)[0]['code'].split(
                    '-')[0],
                outbound_date,
                inbound_date,
                1).find_flights()
            cheapest[name]['city'] = city
            cheapest[name]['temp'] = get_temp(city)
        except Exception:
            print("Exception", name)
            cheapest[name] = {}

        return cheapest

    def on_get(self, req, resp):
        result_json = req.params
        countries = result_json['countries']
        if not isinstance(countries, list):
            countries = countries.split(',')
        print(countries)

        cheapest = {}
        for c in countries:
            cheapest = self.get_flight(c, cheapest)

        resp.body = json.dumps({'cheapest': cheapest})


app.add_route('/login', Login())
app.add_route('/country', Country())
app.add_route('/mappings', Mappings())
app.add_route('/flights', Flights())
