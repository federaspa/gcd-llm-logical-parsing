# compiled_pyke_files.py

from pyke import target_pkg

pyke_version = '1.1.1'
compiler_version = 1
target_pkg_version = 1

try:
    loader = __loader__
except NameError:
    loader = None

def get_target_pkg():
    return target_pkg.target_pkg(__name__, __file__, pyke_version, loader, {
         ('', '.cache_program/', 'facts.kfb'):
           [1740407399.165008, 'facts.fbc'],
         ('', '.cache_program/', 'rules.krb'):
           [1740407399.1716359, 'rules_fc.py'],
        },
        compiler_version)

