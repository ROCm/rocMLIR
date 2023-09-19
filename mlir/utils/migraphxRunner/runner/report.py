from io import StringIO


def process_migraphx_output(filename):
  summary = []
  with open(filename, 'r') as file:
    is_record = False
    for line in file:
      if line:
        if is_record:
          summary.append(line)
        if 'Summary:' in line:
          is_record = True 

  counter = 0
  for line in summary:
    if 'Batch size:' in line:
      break
    counter += 1

  statistics = summary[counter:]
  summary = summary[:counter]

  data = {'Rate': '', 'Total time': '', 'Total instructions time': '', 'Overhead time': ''}
  for key in data.keys():
    for line in statistics:
      if key in line:  
        _, value = line.split(':')
        data[key] = value.strip()
        break

  return data, summary


def make_report_page(group, data):
  file = StringIO()
  file.write(f'## Model: {group.name}\n\n')

  first_model = group.models[0]
  if first_model.is_static():
    file.write(f'**Params** : None\n\n')
  else:
    file.write(f'**Params** : {first_model.params}\n\n')

  for test_type in data.keys():
    if test_type:
      file.write(f'### {test_type.upper()}\n\n')
    else:
      file.write(f'### Native\n\n')

    file.write("| mode | rate | Total time | Total instructions time |\n")
    file.write("| ---- | ---- | ---------- | ----------------------- |\n")
    configs = data[test_type]
    for key in configs.keys():
      statistics = configs[key]['statistics']
      file.write(f'| {key} | {statistics["Rate"]} | {statistics["Total time"]} | {statistics["Total instructions time"]} |\n')

    file.write('\n\n')
    for key in configs.keys():
      summary = configs[key]['summary']
      file.write('<details>\n')
      file.write(f'<summary>{key}</summary>\n\n')
      file.write('```bash\n')
      for line in summary:
        file.write(line)
      file.write('```\n\n')
      file.write('</details>\n\n')

  file.seek(0)
  print(file.read())
