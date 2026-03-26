# Claude Code Instructions

## Jupyter Notebooks — Math Formatting

Always write mathematical expressions in LaTeX when working on Jupyter notebooks (`.ipynb` files):

- **Inline math**: wrap in single dollar signs `$...$`
  - Example: `the selection coefficient $s_d$` or `frequency $f_\alpha(t) \in [0, 1]$`
- **Display math** (equations on their own line): wrap in double dollar signs `$$...$$`
  - Example: `$$\Delta\mathbf{x} = \Sigma\,\mathbf{s}$$`

Never use plain-text code blocks (` ``` `) to write equations in markdown cells. Reserve code blocks for actual Python/code syntax only.

**Common symbols used in this project:**
- Greek letters: `\alpha`, `\beta`, `\gamma`, `\phi`, `\Sigma`, `\sigma`, `\Delta`
- Bold vectors/matrices: `\mathbf{x}`, `\boldsymbol{\phi}`
- Fractions: `\frac{numerator}{denominator}`
- Sums/integrals: `\sum_{\alpha}`, `\int_{t_0}^{T}`
- Sets/spaces: `\mathbb{R}^d`
- Subscripts/superscripts: `x_d`, `x^2`, `t_{k+1}`
