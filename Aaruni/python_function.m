pyenv('Version', 'C:\ProgramData\miniconda3\python.EXE')
py.eval('print("Hello from Python")', py.dict());
py.exec('print("Hello from Python")', py.dict());
sys = py.importlib.import_module('sys');
disp(sys.version)