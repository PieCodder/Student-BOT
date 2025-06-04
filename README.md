# Student-BOT

This repository contains a simple AI trainer script implemented in pure Python. The script allows you to add training examples, train a small model, and test it.

## Usage

1. **Add examples**
   ```bash
   python ai_trainer.py add <x1> <x2> <label>
   ```
   `x1` and `x2` are feature values, and `label` is the expected class (`0` or `1`).

2. **Train the model**
   ```bash
   python ai_trainer.py train
   ```
   This reads all saved examples and creates a model saved in `model.json`.

3. **Test**
   ```bash
   python ai_trainer.py test <x1> <x2>
   ```
   The command loads the trained model and outputs the predicted label and probability.

Training data is stored in `training_data.csv`.

