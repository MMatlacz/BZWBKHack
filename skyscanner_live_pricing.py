import time
from copy import deepcopy
from datetime import datetime, timedelta

import requests

from config import skyscanner_token
from data_provider import DataProvider


class LivePricing:
    def __init__(self, origin, destination, start_date, end_date, adults,
                 market='PL', currency='PLN', locale='en-GB'):
        data = {'apiKey': skyscanner_token, 'country': market,
                'currency': currency,
                'locale': locale, 'originplace': origin + '-sky',
                'destinationplace': destination + '-sky',
                'outbounddate': start_date, 'inbounddate': end_date,
                'adults': adults}
        url = 'http://partners.api.skyscanner.net/apiservices/pricing/v1.0'
        r = requests.post(url, data)
        while r.status_code == '204':
            time.sleep(1)
            r = requests.post(url, data)
        self.get_url = r.headers[
                           'Location'] + '?apiKey=' + skyscanner_token + '&pagesize=1'

    def _get_cheapest(self):
        # time.sleep(5)
        r = requests.get(self.get_url)
        while r.status_code == '204':
            time.sleep(1)
            r = requests.get(self.get_url)
        try:
            json_data = r.json()
        except Exception:
            return []
        return json_data

    @staticmethod
    def _del_keys(d):
        copy = deepcopy(d)
        for key in d.keys():
            if key not in {'Carriers', 'arrival_date', 'arrival_time', 'time',
                           'date', 'Duration'}:
                del copy[key]
        return copy

    def _process_data(self, c, details, agents_mapping, carriers_mapping):
        inbound = details[c['InboundLegId']]
        outbound = details[c['OutboundLegId']]

        inbound['date'] = datetime.strptime(inbound['Departure'],
                                            '%Y-%m-%dT%H:%M:%S').date()
        outbound['date'] = datetime.strptime(outbound['Departure'],
                                             '%Y-%m-%dT%H:%M:%S').date()
        inbound['time'] = datetime.strptime(inbound['Departure'],
                                            '%Y-%m-%dT%H:%M:%S').time()
        outbound['time'] = datetime.strptime(outbound['Departure'],
                                             '%Y-%m-%dT%H:%M:%S').time()

        inbound['arrival_date'] = datetime.strptime(inbound['Arrival'],
                                                    '%Y-%m-%dT%H:%M:%S').date()
        outbound['arrival_date'] = datetime.strptime(outbound['Arrival'],
                                                     '%Y-%m-%dT%H:%M:%S').date()
        inbound['arrival_time'] = datetime.strptime(inbound['Arrival'],
                                                    '%Y-%m-%dT%H:%M:%S').time()
        outbound['arrival_time'] = datetime.strptime(outbound['Arrival'],
                                                     '%Y-%m-%dT%H:%M:%S').time()

        try:
            inbound['Carriers'] = [carriers_mapping[x] for x in
                                   inbound['Carriers']]
        except KeyError:
            pass

        try:
            outbound['Carriers'] = [carriers_mapping[x] for x in
                                    outbound['Carriers']]
        except KeyError:
            pass

        c['InboundDetails'] = self._del_keys(inbound)
        c['OutboundDetails'] = self._del_keys(outbound)

        del c['OutboundLegId']
        del c['InboundLegId']
        del c['BookingDetailsLink']
        for x in c['PricingOptions']:
            x['Agents'] = [agents_mapping[p] for p in x['Agents']]
        return c

    def _parse_data(self, cheapest):
        carriers_mapping = {x['Id']: (x['Name'], x['ImageUrl']) for x in
                            cheapest['Carriers']}
        agents_mapping = {c['Id']: c['ImageUrl'] for c in cheapest['Agents']}
        details = {c['Id']: c for c in cheapest['Legs']}
        results = []
        for c in cheapest['Itineraries']:
            results.append(self._process_data(
                c, details=details,
                agents_mapping=agents_mapping,
                carriers_mapping=carriers_mapping)
            )
        return results

    def find_flights(self):
        """

        :return: List of flights with details
        """
        cheapest = self._get_cheapest()
        cheapest = self._parse_data(cheapest)

        for c in cheapest:
            # c['price'] = sorted(
            #     c['PricingOptions'], key=lambda x: x['Price'])[0]
            c['price'] = c['PricingOptions'][0]
            del c['PricingOptions']

        # cheapest = sorted(cheapest, key=lambda x: x['price']['Price'])
        cheapest = cheapest[0]
        new_cheapest = {'time': str(cheapest['OutboundDetails']['time']),
                        'duration': cheapest['OutboundDetails']['Duration'],
                        'price': cheapest['price']['Price'],
                        'agent_img': cheapest['price']['Agents'][0],
                        'link': cheapest['price']['DeeplinkUrl']}
        del cheapest
        return new_cheapest


if __name__ == '__main__':
    outbound = (datetime.today() + timedelta(days=1)).date()
    inbound = (datetime.today() + timedelta(days=7)).date()

    country = requests.get('https://restcountries.eu/rest/v2/alpha/aut').json()
    # print(DataProvider.get_suggestions(country['capital']))
    # print(DataProvider.get_suggestions('Warszawa Chopina'))
    cheapest = LivePricing(
        DataProvider.get_suggestions('Warszawa Chopina')[0]['code'].split('-')[
            0],
        DataProvider.get_suggestions(country['capital'])[0]['code'].split('-')[
            0],
        outbound,
        inbound,
        1).find_flights()
    print(cheapest)
