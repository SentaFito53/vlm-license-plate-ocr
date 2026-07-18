# Vehicle License Plate OCR using Visual Language Model (VLM)

This program performs Optical Character Recognition (OCR) on Indonesian vehicle license plates using a Visual Language Model (VLM) served through LM Studio and integrated with Python. Predictions are evaluated using the Character Error Rate (CER) metric.

## Dataset

Dataset used: [Indonesian License Plate Recognition Dataset](https://www.kaggle.com/datasets/juanthomaswijaya/indonesian-license-plate-dataset) (`test` folder).

Expected dataset folder structure:
```
Indonesian License Plate Recognition Dataset/
├── images/test/       # license plate images
├── labels/test/       # YOLO-format ground truth labels (per character)
└── classes.names      # character list (class_id -> character mapping)
```

## Requirements

- Python 3.10+
- [LM Studio](https://lmstudio.ai/) (to run the VLM locally)
- A multimodal VLM model, e.g. `llava-llama-3-8b-v1_1`

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/SentaFito53/vlm-license-plate-ocr.git
   cd vlm-license-plate-ocr
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running LM Studio

1. Open the LM Studio application.
2. Download a VLM model (e.g. `llava-llama-3-8b-v1_1`) via the **Discover/Search** menu.
3. Switch to **Developer** mode → select the downloaded model → click **Start Server**.
4. Make sure the server is running at `http://localhost:1234` (default) and keep it open while the program runs.

## Running the Program

Adjust the dataset path in the configuration section of `main.py` if needed, then run:

```
python main.py
```

The program will:
1. Read every license plate image in the `images/test` folder.
2. Reconstruct the ground truth from the per-character YOLO labels in `labels/test`.
3. Send each image to the VLM through LM Studio with an OCR prompt.
4. Clean the model's raw output using regex.
5. Compute the CER (Character Error Rate) for each prediction.
6. Save all results to a CSV file (`hasil_ocr_plat.csv`) with columns: `image, ground_truth, prediction, CER_score`.

## CER Formula

```
CER = (S + D + I) / N
```
- S = number of substituted characters
- D = number of deleted characters
- I = number of inserted characters
- N = number of characters in the ground truth

## Output

The `hasil_ocr_plat.csv` file contains the prediction results and CER score for each image, along with the overall average CER printed to the terminal once the program finishes.

## Repository Structure

```
vlm-license-plate-ocr/
├── main.py              # main program
├── requirements.txt     # Python dependency list
├── README.md            # this documentation
└── .gitignore
```
