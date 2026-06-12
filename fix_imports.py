import glob
import os

inject = "import sys, os\nsys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))\n"

for d in ['rpa_pipeline', 'scripts']:
    for f in glob.glob(d + '/*.py'):
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
        if 'sys.path.insert' not in content:
            with open(f, 'w', encoding='utf-8') as file:
                file.write(inject + content)
