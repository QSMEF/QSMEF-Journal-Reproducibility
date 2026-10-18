PRUEBAS DE REGRESIÓN DE QSMEF

Todos los archivos de este paquete deben permanecer en la misma carpeta.

Windows: ejecutar ejecutar_pruebas.bat.
Alternativa: python -m pip install -r requirements_qsmef.txt
             python -m unittest -v test_qsmef_regression.py

La prueba SKW con evolución modificada comprueba además el valor numérico
phi_O = 0.134201 y la diferencia delta phi_O = 0.387256 (tolerancia 2e-5).

El JSON contiene valores de referencia de una ejecución anterior y no es una
validación externa independiente. Las pruebas no certifican por sí solas
la suficiencia del contrato semántico.

El archivo qsmef_regression_run.log contiene la ejecución comprobada del paquete.
