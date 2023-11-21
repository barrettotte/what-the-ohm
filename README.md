# what-the-ohm

Deep learning model to identify resistor bands.

## General Idea / Limitations

This is my first deep learning project and it didn't really go as I wished.

The original idea was to build a model to identify both 4 and 5 band resistor bands.
Once bands are identified, the resistor value could be calculated using band color and position.

In hindsight, this resistor calculation is difficult because of resistors in different orientations.
For example, given a 4-band resistor with bands yellow, violet, orange, gold; 
What is the first band? Yellow or gold ? How would you be able to know given any arbitrary resistor?

The model also seemed to have trouble with training on both resistor types 
and performed much better when focusing on either 4 or 5 band resistors.

I also probably didn't have enough data or variety in data to train the model very well.
My dataset is pretty imbalanced as seen in [notebooks/explore.ipynb](notebooks/explore.ipynb).

## Usage

train model - `train.py`

see [notebooks/test-model.ipynb](notebooks/test-model.ipynb)

## Dataset

Images of 4 and 5 band resistors taken with USB microscope.

- 67 values of 4 band resistors
- 60 values of 5 band resistors
- 1270 images in total

Dataset prepped and bundled using `dataset.py`.
Dataset should be put in `./data/` to use for training model in `train.py`.

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
- https://stackoverflow.com/questions/65718296/multi-label-classification-with-unbalanced-labels
