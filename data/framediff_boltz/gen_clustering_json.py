import json

clustering = {}
with open("clusters-by-entity-40.txt") as fp:
    for line in fp:
        line = line.strip()
        members = line.split(" ")
        rep = members[0].lower()
        # print(members, rep)
        for entity in members:
            clustering[entity.lower()] = rep

with open("clustering-40.json", 'w') as fp:
    json.dump(clustering, fp)
