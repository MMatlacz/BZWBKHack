import random

import requests


def convert_temp(temp):
    return temp - 273.15


def get_temp(city):
    r = requests.get(
        "http://api.openweathermap.org/data/2.5/weather?q={}&appid=bd5e378503939ddaee76f12ad7a97608".format(
            city))
    data = r.json()
    try:
        temp = data['main']['temp']
        return round(convert_temp(temp), ndigits=1)
    except Exception:
        return random.randint(0, 30)
