labels = [{'name':'bolt', 'id':1}, {'name':'nut', 'id':2}]
label_map = {'item': labels}

with open('label_map.pbtxt', 'w') as f:
    for label in labels:
        f.write('item {\n')
        f.write('  id: {}\n'.format(label['id']))
        f.write('  name: "{}"\n'.format(label['name']))
        f.write('}\n')