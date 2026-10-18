# QSMEF — Functional Contribution Analysis in Quantum Circuits

**QSMEF** (*Quantum Software Engineering Module Evaluation Framework*) is a framework for evaluating the functional contributions of blocks in quantum circuits. It uses coalition-based observable measurements and Shapley values, subject to explicitly declared technical, functional, and semantic applicability conditions.

This repository accompanies the QSMEF journal manuscript and makes available the computational implementation, case-study artifacts, regression tests, and supporting literature-review materials.

## Research objective

Evaluating a quantum circuit as a whole does not directly explain how its functional blocks contribute to a selected observable property. QSMEF addresses this attribution question by treating blocks as players in a cooperative game, evaluating configurations in which omitted blocks are neutralized by identity, and calculating Shapley values **when the declared applicability conditions are met**.

The framework distinguishes four validation outcomes:

- `APPLICABLE`
- `NOT APPLICABLE UNDER THE DECLARED SCOPE`
- `SEMANTIC CONTRACT REQUIRED`
- `SPECIFICATION ERROR`

**Interpretation:** An `APPLICABLE` verdict means that the implemented checks and researcher-declared criteria are satisfied for the specified configuration. It is not an independent proof of complete functional or semantic comparability. Negative results under a declared scope should not be generalized to all configurations.

## Repository contents

```text
QSMEF-Revista-2026/
├── README.md                         # Repository overview (this file)
├── QSMEF-JOURNAL/
│   ├── README.md                     # Computational documentation and instructions
│   └── resultados/
│       ├── QSMEF_Implementation.py
│       ├── test_qsmef_regression.py
│       ├── qsmef_checks.json
│       ├── requirements_qsmef.txt
│       ├── ejecutar_pruebas.bat
│       ├── README_pruebas_QSMEF.txt
│       ├── qsmef_regression_run.log
│       ├── experiment_summary.csv
│       └── verification_results.txt
└── revisión_literatura/
    └── datos/
        ├── retrieved_studies.csv
        ├── excluded_studies.csv
        └── selected_studies_analysis.xlsm
```

The `QSMEF-JOURNAL/` directory contains the computational materials associated with the manuscript. The `revisión_literatura/` directory contains supporting bibliographic datasets and analysis files; it is separate from the computational regression suite.

For methodological details, requirements, experimental interpretation, and test instructions, see the **[computational README](QSMEF-JOURNAL/README.md)**.

## Included computational case studies

1. **QAOA–MaxCut:** cycle graph C6, depth `p = 2`.
2. **TwoLocal–H₂:** four-qubit Jordan–Wigner representation, with a two-electron-sector restriction.
3. **SKW quantum walk:** hypercube search, with oracle, coin, and shift blocks.
4. **QPE–Shor:** quantum phase estimation associated with modular period finding.
5. **Graph partitioning / minimum bisection:** fixed Hamming-weight sector and XY mixer.

The SKW experiments also include two distinct modified-oracle comparisons: a **full-evolution comparison** and an auxiliary **fixed-state comparison**. Their results should not be interpreted interchangeably.

## Quick start

Install Python and open a terminal in `QSMEF-JOURNAL/resultados/`:

```bash
python -m pip install -r requirements_qsmef.txt
python QSMEF_Implementation.py
```

Run the nine regression tests from the same directory:

```bash
python -m unittest -v test_qsmef_regression.py
```

On Windows, `ejecutar_pruebas.bat` provides an alternative way to launch the tests after cloning or extracting the repository.

The JSON reference values and recorded logs support computational traceability. Passing the regression tests checks consistency with previously recorded outputs; it does **not** constitute independent experimental or semantic validation.

## Reproducibility and limitations

Attributions are specific to the circuit configuration, initial state, observable, functional decomposition, neutralization rule, and declared applicability criteria. Executable semantic contracts test formalized conditions chosen by the researcher; they do not establish that every possible interpretation of functional comparability has been verified. Numerical results may vary slightly with software versions and numerical optimization settings.

For further details, consult [QSMEF-JOURNAL/README.md](QSMEF-JOURNAL/README.md).

## Publication and citation

This repository supports the QSMEF journal manuscript. The bibliographic reference for the final article and the **version-specific Zenodo DOI** will be added once the corresponding publication and software archive are finalized.

When citing the software, use the Zenodo DOI for the **specific archived release** used to obtain the reported results. Do not assume that a DOI from an earlier version identifies the files in the current repository.
