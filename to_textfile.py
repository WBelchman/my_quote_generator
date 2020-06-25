#There were a lot of duplicates so... this exists now

import json
from collections import OrderedDict

f = open("quotes.json", "r", encoding='utf8')
data = json.load(f)

quotes = []
for x in data:
    quotes.append(x["Quote"]+"\n")

f2 = open("quotes.txt", "x", encoding='utf8')

new_quotes = list(OrderedDict.fromkeys(quotes)) 

f2.writelines(new_quotes)