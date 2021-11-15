# FastTransformer - Easy Transformers for Text Classification


FastTransformer is an easy-to-use Python library for training transformers for text classification tasks. It takes care of most of the intricacies frequently faced in practice and only leaves the most fundamental settings of the classification problem for the user to define.

## Installation

FastTransformer requires **Python version >= 3.6**. It can be installed cloning the repo from the command line:

```shell
git clone https://github.com/julianregatky/fasttransformer
```

Make sure to have all modules and their dependencies installed

```shell
pip install -r requirements.txt
```

## Quickstart

* Import the `FastTransformer` class from `fasttransformer` and create an instance of the class `FastTransformer`

```python
from fasttransformer import FastTransformer

ft = FastTransformer(
    pretrained_path='bert-base-cased',
    num_labels=2,
    do_lower_case=False,
    batch_size=32,
    max_length=30,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    output_dir='trained_model'
)
```

* If you have unlabeled data or have downsampled your majority class, you can fine-tune your pre-trained model on masked language

```python
ft.pretrain_mlm(
    list(train.text.values),
    epochs=2,
    mlm_probability=0.15,
    output_dir='pretrained_mlm',
    random_state=42,
    update_classifier_pretrained_model=True
)
```

* Initialize the optimizer, build your training dataloader and train your classification model in just three lines

```python
ft.set_optimizer()
train_dataloader = ft.transform(
    train.text.values,
    train.label.values,
    sampler='random'
)
ft.train_classifier(train_dataloader,
    epochs=5,
    random_state=42
)
```

* Obtain quick predictions

```python
test_dataloader = ft.transform(
    test.text.values,
    sampler='sequential'
)
preds = ft.predict(test_dataloader)
```

## Documentation

Please find the API documentation [here](https://raspy-pet-2ee.notion.site/FastTransformer-5433308fce8f4eaca137453d15d633fc) (coming soon).

Also check out the example notebook, available in [colab](https://colab.research.google.com/drive/1Od9z7zZFtwXyXP0bRuyVWtK0HH03f07R?usp=sharing).

## Improvements to be incorporated ASAP

* API Docs
* Better printing of training and validation loss
* More complete sample dataset

## License

[MIT License](LICENSE)
