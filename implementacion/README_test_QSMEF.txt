QSMEF REGRESSION TESTS

All files in this package must remain in the same directory.

Windows: Run ejecutar_pruebas.bat.
Alternative: python -m pip install -r requirements_qsmef.txt
             python -m unittest -v test_qsmef_regression.py

The SKW test with modified evolution also verifies the numerical value
phi_O = 0.134201 and the difference delta phi_O = 0.387256
(tolerance 2e-5).

The JSON file contains reference values from a previous execution and
does not constitute independent external validation. The tests alone
do not establish the sufficiency of the semantic contract.

The file qsmef_regression_run.log contains the verified execution
record of the test suite.
