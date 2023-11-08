# what-the-ohm

Deep learning model to identify resistor values

## Usage

TODO:

## Dataset

Images of 4 and 5 band resistors taken with USB microscope.

- 67 values of 4 band resistors
- 60 values of 5 band resistors
- 1270 images in total

Dataset prepped and bundled using `dataset.py`.

Finalized dataset hosted at https://www.kaggle.com/datasets/barrettotte/resistors

## Development

```sh
# init environment
make env_init
conda activate what-the-ohm
```

## References

- https://pytorch.org/docs/stable/index.html
- https://www.digikey.com/en/resources/conversion-calculators/conversion-calculator-resistor-color-code
