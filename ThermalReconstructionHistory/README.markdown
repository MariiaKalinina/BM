# Thermal History Reconstruction

This repository contains a Python script for modeling thermal history reconstruction of lithospheric processes, including heat flow, temperature profiles, and thermal subsidence based on the McKenzie model. The script reads parameters from a JSON file and generates plots to visualize the results.

## Prerequisites

- Python 3.6+
- Required libraries: `numpy`, `scipy`, `matplotlib`, `pandas`
- Tkinter for GUI display (included with Python, or install `python3-tk` on Linux)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MariiaKalinina/BM.git
   cd BM/ThermalReconstructionHistory
   ```

2. Install dependencies:
   ```bash
   pip install numpy scipy matplotlib pandas
   ```

3. (Optional) On Linux, ensure Tkinter is installed:
   ```bash
   sudo apt-get install python3-tk
   ```

## Usage

1. Ensure `inputdata.json` is in the `ThermalReconstructionHistory` directory with the required parameters (see `inputdata.json` example below).
2. Run the script:
   ```bash
   python thermal_history_reconstruction.py inputdata.json
   ```

## Input File

Create an `inputdata.json` file with the following parameters:

```json
{
    "a": 125000.0,
    "t_c": 30000.0,
    "kappa": 1.234e-6,
    "k": 2.6,
    "T_m": 1330.0,
    "alpha": 3.28e-5,
    "rho_m": 3330.0,
    "rho_c": 2800.0,
    "rho_w": 1030.0,
    "beta_c": 1.2,
    "beta_sc": 1.2,
    "G_prime": 10.0,
    "duration_myr": 200.0,
    "A0": 1.0e-6,
    "a_r": 10000.0,
    "rho_r": 2700.0,
    "U": 2.0,
    "Th": 8.0,
    "K": 1.5
}
```

## Output

- **Console output**: Parameters, lithosphere thickness calculations, and confirmation of plot saving.
- **Plots**: Three subplots (heat flow evolution, temperature profile, thermal subsidence) saved as `thermal_history_plots.png` and displayed in a GUI window.

## Notes

- Uses Matplotlib's `TkAgg` backend for plot display. If no GUI is available, modify the script to use `Agg` backend for saving plots without display.
- Ensure `inputdata.json` is correctly formatted to avoid errors.
- For Jupyter users, run with:
  ```python
  %cd BM/ThermalReconstructionHistory
  !python thermal_history_reconstruction.py inputdata.json
  ```

## License

MIT License