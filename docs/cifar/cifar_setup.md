Thanks for confirming! I’ll dig into the `tachyon-beep/kaslite` repo and develop a detailed run sheet showing how to download and test the CIFAR dataset using the morphogenetic experiment setup. I’ll include any necessary code modifications and make sure it's structured to support extensibility for other datasets too. I’ll get back to you shortly with everything laid out clearly.

# Morphogenetic Experiment with CIFAR-10 in Kaslite

This guide shows how to run the **Kaslite** morphogenetic experiment on the CIFAR-10 image dataset. We cover environment setup, code changes to support CIFAR, and example commands to execute the experiment. The Kaslite codebase uses a CLI `--problem_type` flag to select synthetic datasets (spirals, moons, etc); we will add support for CIFAR-10.

## 1. Setup Environment

* **Clone the repo:** Get the latest Kaslite code.

  ```bash
  git clone https://github.com/tachyon-beep/kaslite.git
  cd kaslite
  ```

* **Install dependencies:** The project requires PyTorch, NumPy, and scikit-learn as specified in `pyproject.toml`. Install these plus `torchvision` (needed for CIFAR-10):

  ```bash
  pip install torch numpy scikit-learn torchvision
  ```

  *Note:* The `pyproject.toml` lists `torch>=1.9.0`, `numpy>=1.20.0`, `scikit-learn>=1.0.0`. You should also install a compatible version of `torchvision` for CIFAR support.

* **GPU (optional):** If you have CUDA available, ensure PyTorch is installed with CUDA support so you can use `--device cuda` for faster training.

## 2. Prepare the CIFAR-10 Dataset

* **Download CIFAR-10:** Use PyTorch’s `torchvision.datasets` to download CIFAR-10 into a local folder. For example, in Python:

  ```python
  from torchvision.datasets import CIFAR10
  dataset = CIFAR10(root='data/cifar', train=True, download=True)
  ```

  This will fetch the training set (50,000 images) and labels. By default, CIFAR-10 images are 32×32 RGB.

* **Preprocess images:** We will flatten each image into a 3072‑dimensional vector (`32×32×3 = 3072`). The Kaslite code uses `StandardScaler` on numeric features; for images you can either apply a similar normalization (per-pixel mean=0, variance=1) or use a standard image transform (e.g. scaling to \[0,1] or using channel means). In code, you might convert the downloaded images as follows:

  ```python
  import numpy as np
  X = dataset.data.astype(np.float32)  # shape (50000, 32, 32, 3)
  X = X.reshape(len(X), -1)            # shape (50000, 3072)
  y = np.array(dataset.targets)       # shape (50000,)
  # Optionally scale X, e.g. using StandardScaler or divide by 255
  ```

* **Train/validation split:** Kaslite splits data by a `--train_frac` parameter. For CIFAR, either split the 50,000 images (e.g. 80/20) or use the provided test set. To follow the existing pattern, you can treat the 50,000 CIFAR training images as one pool and let `get_dataloaders()` do the split (using `torch.utils.data.random_split`).

## 3. Code Changes for CIFAR Compatibility

The Kaslite script `scripts/run_morphogenetic_experiment.py` must be modified to handle CIFAR:

* **Add a `cifar10` problem type:** In the argument parser, include “cifar10” (or similar) as a valid `--problem_type`. Currently, the choices are hard-coded to `["spirals", "moons", "clusters", "spheres", "complex_moons"]`. For example, change that line to:

  ```python
  parser.add_argument("--problem_type",
      choices=["spirals", "moons", "clusters", "spheres", "complex_moons", "cifar10"],
      default="spirals",
      help="Type of problem to solve"
  )
  ```

  This lets you run the script with `--problem_type cifar10`.

* **Update `get_dataloaders` dispatch:** In the function `get_dataloaders(args)`, add a new branch for the CIFAR-10 case. Currently the code checks `args.problem_type` and calls a generator like `create_spirals` or `create_moons`. For CIFAR-10, insert something like:

  ```python
  elif args.problem_type == "cifar10":
      from torchvision.datasets import CIFAR10
      from sklearn.preprocessing import StandardScaler

      # Load CIFAR-10 train set
      dataset = CIFAR10(root="data/cifar", train=True, download=True)
      X = dataset.data.astype(np.float32).reshape(len(dataset), -1)  # Flatten images
      y = np.array(dataset.targets, dtype=np.int64)

      # (Optional) Normalize pixel values, e.g. divide by 255 or use StandardScaler
      scaler = StandardScaler().fit(X)
      X = scaler.transform(X).astype(np.float32)
  ```

  After this branch, the code will continue to split into train/val loaders as usual (the existing scaler and `TensorDataset` code applies to any numeric `X, y`). This way, `train_loader, val_loader` are returned just like for synthetic data.

* **Set `input_dim` appropriately:** Since images are flattened, set `args.input_dim = 3072` (32×32×3). You can do this when invoking the script via CLI (see next section) or override in code. The `BaseNet` model uses `input_dim` to size the first `nn.Linear` layer. For example:

  ```bash
  python scripts/run_morphogenetic_experiment.py --problem_type cifar10 --input_dim 3072 ...
  ```

* **Adjust the model’s output layer:** By default, `BaseNet` ends with `self.out = nn.Linear(hidden_dim, 2)`, i.e. a 2-class output (since all example problems are binary). For CIFAR-10, change this to 10 outputs. You can either modify the code in `BaseNet.__init__`, or after construction do:

  ```python
  model = BaseNet(args.hidden_dim, seed_manager, input_dim=args.input_dim, ...)
  model.out = nn.Linear(args.hidden_dim, 10)  # 10 classes for CIFAR-10
  ```

  Also ensure the loss function matches 10 classes (the script uses `nn.CrossEntropyLoss`, which is fine). This step is crucial because otherwise the network would not predict 10 classes.

* **Other parameters:** You may want to increase `hidden_dim` (the default 128 might be small for images) and adjust batch size/learning rate. For example, using `--hidden_dim 512 --batch_size 128` can help learn from image data.

These code changes enable the CIFAR-10 data path. The key modifications are adding the `cifar10` case in argument parsing and data loading, and changing the final layer to 10 classes.

## 4. Running the Experiment

With the above changes, you can run the morphogenetic training on CIFAR-10. A sample workflow is:

1. **Set up logging directory:** The script writes logs under `results/`. Ensure you have write access; the code will create `results` if needed.

2. **Execute the script:** For example:

   ```bash
   python scripts/run_morphogenetic_experiment.py \
       --problem_type cifar10 \
       --input_dim 3072 \
       --hidden_dim 512 \
       --blend_steps 200 \
       --shadow_lr 0.001 \
       --batch_size 128 \
       --train_frac 0.8 \
       --device cuda
   ```

   This command specifies the CIFAR-10 problem, sets the flattened input dimension, a larger hidden layer, and uses GPU. Adjust `--blend_steps`, `--shadow_lr` etc. as needed. The script will download CIFAR-10 automatically (if not present) and begin training. It will log progress to the console and to a JSON file in `results/`.

3. **Monitor output:** The `ExperimentLogger` prints epoch metrics and seed events in real time. Results (epoch losses, accuracy, seed state changes) are also saved in `results/<timestamp>.json` and in a CSV log file for seed states. At the end, a summary of training and adaptation phases will be printed.

## 5. Extensibility and Notes on Dataset Support

The Kaslite code is structured so that new datasets can be added by extending the `get_dataloaders` dispatch and corresponding CLI choices. For future datasets, you would:

* **Define a new case** in `--problem_type` and in `get_dataloaders`, either generating data or loading from a dataset class.
* **Ensure input/output dims match:** For any new dataset, set `input_dim` to the feature length and adjust the model’s output layer to the number of classes or targets.
* **Normalization:** The current pipeline uses `StandardScaler` on numeric features. For other data types (images, text embeddings, etc.), consider using appropriate preprocessing (e.g. PyTorch transforms) instead.

To **preserve and enhance extensibility**, one could refactor `get_dataloaders` into a registry of dataset loaders, or use a configuration file to specify dataset paths and transforms. For example, employing PyTorch `Dataset` classes directly (as we did for CIFAR-10) helps decouple data loading from model code. Keeping the model modular (as Kaslite does with the `BaseNet` interface) means the same architecture can be reused for any fixed-dimensional input, as long as `input_dim` and final layer are set appropriately.

In summary, by adding a new `problem_type` branch and handling the image data correctly, you can run the morphogenetic experiment on CIFAR-10. The code’s design (with a clear CLI and `get_dataloaders` switch) makes adding datasets straightforward, but be mindful to match the neural net’s dimensions (e.g. change the final `out` layer from 2 to 10 for CIFAR).

**References:** Kaslite codebase (argument parsing and data loading); BaseNet output layer definition; project dependencies in `pyproject.toml`.
