# ADL_FINAL

## Prerequisite

1. `keras`
2. `sklearn`
3. `PIL`
4. `pandas`
5. `matplotlib`

## Data



## Personal Model

### How to run:

```sh
# please use python3 if your default python is of version 2
python PersonalModel.py <person>
```

The `<person>` parameter should be the name of the directory in which the person's data is.
For example, `python PersonalModel.py 12_Te_Yan_Wu` will train a personal model for Te-Yan Wu, one of our admirable voluntary participants.

### Model Structure

The structure of Personal Model is as follows:

```python
# Firstly, a convolutional layer:
Conv2D
MaxPooling2D
Dropout

# And then a second convolutional layer:
Conv2D
MaxPooling2D
Dropout

# flatten
Flatten

# fully-connected
Dense
Dropout

# two more fully-connected layers
Dense
Dense
```
