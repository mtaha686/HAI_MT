# Colab Training Bundle

This folder contains only the minimal data and code needed to train the model on Google Colab. Zip this folder and upload it to Colab, train, then download the resulting `model/` back to your project.

## Usage (Local)

1. Copy dataset files from your project into this folder:
```
python prepare_data.py
```
This copies from `../data/train.jsonl` and `../data/test.jsonl` into `colab_train/data/`.

2. Zip this folder for upload:
```
cd ..
tar -czf colab_train.tgz colab_train
# or on Windows Git Bash:
zip -r colab_train.zip colab_train
```

## Usage (Colab)

1. Upload the zip and extract:
```
%cd /content
from google.colab import files
up = files.upload()  # upload colab_train.zip
!unzip -o colab_train.zip -d /content
%cd /content/colab_train
```

2. Install dependencies and check GPU:
```
!pip -q install -r requirements.txt
!nvidia-smi
```

3. Train (GPU if available):
```
!python train_colab.py \
  --model distilgpt2 \
  --train-data data/train.jsonl \
  --eval-data data/test.jsonl \
  --output-dir model \
  --epochs 5 \
  --batch-size 8 \
  --learning-rate 5e-5 \
  --max-length 512 \
  --use-lora \
  --device auto
```

If you hit OOM, reduce `--batch-size` or `--max-length` (e.g., 256).

4. Download the trained model back:
```
from google.colab import files
!zip -r model_zip model
files.download('model_zip.zip')
```

## After Downloading

Place the `model/` (unzipped) into your project root (e.g., `herbal_model/`), then run your API locally:
```
python src/api.py --model_path herbal_model
```



