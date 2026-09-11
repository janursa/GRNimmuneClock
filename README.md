# GRNimmuneClock

**Cell-Type Specific Aging Clocks for Immune Cells**

GRNimmuneClock provides pre-trained aging clocks for immune cell types, trained on genes significantly associated with age. Predict biological age from gene expression data with cell-type specific models trained on multiple cohorts. Each clock also ships with a consensus gene regulatory network (GRN) for the same cell type, usable to interpret the clock's genes in terms of transcription factor (TF) activity.

## Features

- 🔬 **Cell-Type Specific**: Separate models for CD4T and CD8T cells
- 🧬 **Age-Associated Features**: Trained on genes significantly correlated with age, not a fixed gene panel
- 🔗 **Network Analysis**: Access bundled consensus GRNs for TF-target exploration, and score TF activity from a clock's coefficients (`tf_activity_from_coefs`)
- 🔍 **Interpretation**: Regulon overrepresentation among a clock's genes (`regulon_ora`)
- 🎨 **Visualization Tools**: Built-in plotting functions for analysis
- 🚀 **Easy to Use**: Simple Python API
- 🔧 **Training Pipeline**: Tools to train custom aging clocks

## Installation

```bash
pip install grnimmuneclock
```

Or install from source:

```bash
git clone https://github.com/janursa/GRNimmuneClock.git
cd GRNimmuneClock
pip install -e .
```

## Quick Start

```python
from grnimmuneclock import AgingClock, load_example_data
import grnimmuneclock.plotting as gplot

# Load pre-trained clock for CD4T cells
clock = AgingClock(cell_type='CD4T')

# Load example data
adata = load_example_data()

# Predict biological age
adata_predicted = clock.predict(adata)
print(adata_predicted.obs['predicted_age'])

# Visualize predictions
gplot.plot_predicted_vs_actual(adata_predicted, hue='sex')
```
See the tutorial.ipynb for more.

## Supported Cell Types

- `CD4T`: CD4+ T cells 
- `CD8T`: CD8+ T cells 


## Model Information

All models are:
- **Algorithm**: Ridge regression with StandardScaler
- **Features**: Gene expression values, restricted per cell type to genes significantly associated with age (not a GRN target list)
- **Training**: Multiple cohorts (European, Korean, Japanese, Chinese)
- **Age Range**: 20-80 years
- **Species**: Human
- **Tissue**: Peripheral blood

Per-model feature counts and held-out performance are in `grnimmuneclock/models/<cell_type>/metadata.json`.


## Citation

If you use GRNimmuneClock in your research, please cite:

```bibtex
@article{nourisa2025grnimmuneclock,
  title={TBD},
  author={Nourisa, Jalil and others},
  journal={TBD},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For questions and issues, please open an issue on [GitHub](https://github.com/janursa/GRNimmuneClock/issues).

