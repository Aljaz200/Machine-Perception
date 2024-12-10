from nbconvert import PythonExporter

# Pot do notebooka
notebook_filename = 'ass4.ipynb'

# Ustvarite PythonExporter
exporter = PythonExporter()

# Izvozi notebook kot Python skripto
body, resources = exporter.from_filename(notebook_filename)

# Shranite Python skripto
with open('ass4.py', 'w') as f:
    f.write(body)
