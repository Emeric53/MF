import os
script_dir = os.path.dirname(os.path.abspath(__file__))

testfile = 'wavelength.txt'
print(os.path.isfile(os.path.join(script_dir,testfile)))
print(os.path.join(script_dir,testfile))
