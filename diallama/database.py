import json
import random
import copy
import os
import re
from typing import Text, Dict

from fuzzywuzzy import fuzz
from word2number import w2n


class MultiWOZDatabase:
    """ MultiWOZ database implementation. """

    IGNORE_VALUES = {
        'hospital': ['id'],
        'police': ['id'],
        'attraction': ['location', 'openhours'],
        'hotel': ['location', 'price'],
        'restaurant': ['location', 'introduction']
    }

    FUZZY_KEYS = {
        'hospital': {'department'},
        'hotel': {'name'},
        'attraction': {'name'},
        'restaurant': {'name', 'food'},
        'bus': {'departure', 'destination'},
        'train': {'departure', 'destination'},
        'police': {'name'}
    }

    DOMAINS = [
        'restaurant',
        'hotel',
        'attraction',
        'train',
        'taxi',
        'police',
        'hospital'
    ]

    def __init__(self):
        self.data, self.data_keys = self._load_data()

    def _load_data(self):
        database_data = {}
        database_keys = {}

        for domain in self.DOMAINS:
            with open(os.path.join(os.path.dirname(__file__), "database", f"{domain}_db.json"), "r") as f:
                for line in f:
                    if not line.startswith('##') and line.strip() != "":
                        f.seek(0)
                        break
                database_data[domain] = json.load(f)

            if domain in self.IGNORE_VALUES:
                for i in database_data[domain]:
                    for ignore in self.IGNORE_VALUES[domain]:
                        if ignore in i:
                            i.pop(ignore)

            database_keys[domain] = set()
            if domain == 'taxi':
                database_data[domain] = {k.lower(): v for k, v in database_data[domain].items()}
                database_keys[domain].update([k.lower() for k in database_data[domain].keys()])
            else:
                for i, database_item in enumerate(database_data[domain]):
                    database_data[domain][i] = {k.lower(): v for k, v in database_item.items()}
                    database_keys[domain].update([k.lower() for k in database_item.keys()])

        return database_data, database_keys

    def time_str_to_minutes(self, time_string) -> Text:
        """ Converts time to the only format supported by database, i.e. HH:MM in 24h format
            For example: "noon" -> 12:00
        """
        converted_time_string = time_string
        if time_string == "noon":
            converted_time_string = "12:00"
        elif time_string == "midnight":
            converted_time_string = "00:00"
        elif time_string == "morning":
            converted_time_string = "08:00"
        elif time_string == "afternoon":
            converted_time_string = "14:00"
        elif time_string == "evening":
            converted_time_string = "19:00"
        elif time_string == "night":
            converted_time_string = "22:00"
        else:
            time_string = re.sub(r'^after\s*', '', time_string)
            hour, minute = (time_string + ' ').split(' ', 1)
            hour = w2n.word_to_num(hour)
            if minute.strip():
                minute = w2n.word_to_num(minute)
            else:
                minute = 0
                if 'half' in time_string:
                    minute = 30
                elif 'quarter past' in time_string:
                    minute = 15
                elif 'quarter to' in time_string:
                    minute = 45

            if re.search(r'\ba\.?m\.?', time_string):  # am
                hour = hour if hour != 12 else 0
            elif re.search(r'\bp\.?m\.?', time_string):  # pm
                hour = hour + 12 if hour != 12 else 12
            converted_time_string = f"{hour:02d}:{minute:02d}"

        return converted_time_string

    def query(self,
              domain: Text,
              constraints: Dict[Text, Text],
              fuzzy_ratio: int = 90):
        """
        Returns the list of entities (dictionaries) for a given domain based on the annotation of the belief state.

        Arguments:
            domain:      Name of the queried domain.
            constraints: Hard constraints to the query results.
        """

        if domain == 'taxi':
            c, t, p = None, None, None

            c = constraints.get('color', [random.choice(self.data[domain]['taxi_colors'])])[0]
            t = constraints.get('type', [random.choice(self.data[domain]['taxi_types'])])[0]
            p = constraints.get('phone', [''.join([str(random.randint(1, 9)) for _ in range(11)])])[0]

            return [{'color': c, 'type': t, 'phone': p}]

        elif domain == 'hospital':

            hospital = {
                'hospital phone': '01223245151',
                'address': 'Hills Rd, Cambridge',
                'postcode': 'CB20QQ',
                'name': 'Addenbrookes'
            }

            departments = [x.strip().lower() for x in constraints.get('department', [])]
            phones = [x.strip().lower() for x in constraints.get('phone', [])]

            if len(departments) == 0 and len(phones) == 0:
                return [dict(hospital)]
            else:
                results = []
                for i in self.data[domain]:
                    if 'department' in self.FUZZY_KEYS[domain]:
                        f = (lambda x: fuzz.partial_ratio(i['department'].lower(), x) > fuzzy_ratio)
                    else:
                        f = (lambda x: i['department'].lower() == x)

                    if any(f(x) for x in departments) and \
                       (len(phones) == 0 or any(i['phone'] == p.strip() for p in phones)):
                        results.append(dict(i))
                        results[-1].update(hospital)

                return results

        else:
            # Hotel database keys:      address, area, name, phone, postcode, pricerange, type, internet, parking, stars, takesbookings (other are ignored)
            # Attraction database keys: address, area, name, phone, postcode, pricerange, type, entrance fee (other are ignored)
            # Restaurant database keys: address, area, name, phone, postcode, pricerange, type, food

            # Train database contains keys: arriveby, departure, day, leaveat, destination, trainid, price, duration
            # The keys arriveby, leaveat expect a time format such as 8:45 for 8:45 am

            results = []
            query = {}

            if domain == 'attraction' and 'entrancefee' in constraints:
                constraints['entrance fee'] = constraints.pop(['entrancefee'])

            for key in self.data_keys[domain]:
                query[key] = constraints.get(key, [])
                if len(query[key]) > 0 and key in ['arriveby', 'leaveat']:
                    if isinstance(query[key][0], str):
                        query[key] = [query[key]]
                    query[key] = [self.time_str_to_minutes(x) for x in query[key]]
                    query[key] = list(set(query[key]))

            for i, item in enumerate(self.data[domain]):
                for k, v in query.items():
                    if len(v) == 0 or item[k] == '?':
                        continue

                    # accept item[k] if it's earlier than the times in the query
                    if k == 'arriveby':
                        for t in v:
                            if item[k] != ":" and item[k] < t:
                                break

                    # accept item[k] if it's later than the times in the query
                    elif k == 'leaveat':
                        for t in v:
                            if item[k] != ":" and item[k] > t:
                                break

                    # accept item[k] if it matches to the values in query
                    # using fuzzy matching (see `partial_ratio` method in the fuzzywuzzy library
                    else:
                        if isinstance(v, str):  # make sure we are processing a list of lowercase strings
                            v = [v.strip().lower()]
                        else:
                            v = [x.strip().lower() for x in v]

                        if k in self.FUZZY_KEYS[domain]:
                            f = (lambda x: fuzz.partial_ratio(item[k].lower(), x) > fuzzy_ratio)
                        else:
                            f = (lambda x: item[k].lower() == x)
                        if not any(f(x) for x in v):
                            break

                else:  # This gets executed iff the above loop is not terminated
                    result = copy.deepcopy(item)
                    if domain in ['train', 'hotel', 'restaurant']:
                        ref = constraints.get('ref', [])
                        result['ref'] = '{0:08d}'.format(i) if len(ref) == 0 else ref

                    results.append(result)

            if domain == 'attraction':
                for result in results:
                    result['entrancefee'] = result.pop('entrance fee')

            return results
