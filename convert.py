from nbconvert import PythonExporter

# ukaz v cmd: jupyter nbconvert --to script ***.ipynb

# Pot do notebooka
notebook_filename = input('Vnesite pot do notebooka: ')

# Ustvarite PythonExporter
exporter = PythonExporter()

# Izvozi notebook kot Python skripto
body, resources = exporter.from_filename(notebook_filename)

python_filename = notebook_filename.replace('.ipynb', '.py')

# Shranite Python skripto
with open(python_filename, 'w') as f:
    f.write(body)

print(f'Python skripta je bila shranjena kot {python_filename}')
