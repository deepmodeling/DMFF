test: test_admp test_classical test_common test_difftraj test_dimer test_frontend test_mbar test_sgnn test_energy test_utils

test_admp:
	pytest --disable-warnings tests/test_admp

test_classical:
	pytest --disable-warnings tests/test_classical
	
test_common:
	pytest --disable-warnings tests/test_common
   
test_difftraj:
	pytest --disable-warnings tests/test_difftraj
   
test_dimer:
	pytest --disable-warnings tests/test_dimer

test_frontend:
	pytest --disable-warnings tests/test_frontend
	
test_mbar:
	pytest --disable-warnings tests/test_mbar
	
test_sgnn:
	pytest --disable-warnings tests/test_sgnn
	
test_energy:
	pytest --disable-warnings tests/test_energy.py
	
test_utils:
	pytest --disable-warnings tests/test_utils.py
