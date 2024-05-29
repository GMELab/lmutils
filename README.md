# lmutils

A collection of utilities for working with statistical models, particularly in the vein of MonsterLM and RARity.

## Installation

Requires the [GNU Scientific Library](https://www.gnu.org/software/gsl/). Helpful installation instructions for GSL can be found [here](https://docs.rs/GSL/7.0.0/rgsl/#installation).

```bash
cargo add lmutils
```

## R Support

A collection of R functions are provided in the [lmutils.r](https://github.com/mrvillage/lmutils.r) package.

Please note that the R functionality integrated directly into the `lmutils` crate will likely not work when not called from R.
