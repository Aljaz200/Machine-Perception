from nbconvert import PythonExporter
import os

# ukaz v cmd: jupyter nbconvert --to script ***.ipynb

# Pot do notebooka
notebook_filename = input('Vnesite pot do notebooka: ')

# Ustvarite PythonExporter
exporter = PythonExporter()

if not notebook_filename.__contains__('.'):
    notebook_filename += '.ipynb'
elif notebook_filename[-6:] != '.ipynb':
    raise ValueError('Datoteka ni v formatu .ipynb')

if not notebook_filename or not os.path.exists(notebook_filename):
    print(notebook_filename)
    raise ValueError('Datoteka ne obstaja')

# Izvozi notebook kot Python skripto
body, resources = exporter.from_filename(notebook_filename)


python_filename = notebook_filename.replace('.ipynb', '.py')

# Shranite Python skripto
with open(python_filename, 'w') as f:
    f.write(body)

print(f'Python skripta je bila shranjena kot {python_filename}')
