# Changelog

All notable changes to **tvccointreg** are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/), and the
project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.1] - 2026-06-05

### Fixed
- `plot_coint_heatmap` no longer crashes when the model index is non-numeric
  (e.g. quarterly period labels like `1959Q2`).

### Changed
- All plots (`plot_coefficients`, `plot_decomposition`, `plot_fit`,
  `plot_coint_heatmap`) now render a clean, sparse time axis for string/date
  indices instead of cramming hundreds of tick labels.

### Added
- `examples/consumption_multivariate.py`: a two-regressor US consumption
  function (income + real interest rate) on real `statsmodels` macrodata.
- Documentation website (`docs/`) for GitHub Pages.

## [0.1.0] - 2026-06-05

### Added
- Initial release.
- `TVCModel` / `TVCResults`: time-varying-coefficient regression via
  iteratively rescaled GLS (Hildreth-Houck-Swamy variance components).
- `DriverSpec`: three-set coefficient-driver decomposition
  (bias-free / omitted-variable / measurement-error).
- Generalized cointegration tests (Wald + average-effect) with standard
  inference, after Hall, Swamy & Tavlas (2015).
- Journal-quality tables (text / LaTeX / HTML) and Parula-default plots.
- Synthetic data generators and the US consumption-function example.

[0.1.1]: https://github.com/merwanroudane/tvccointreg/releases/tag/v0.1.1
[0.1.0]: https://github.com/merwanroudane/tvccointreg/releases/tag/v0.1.0
