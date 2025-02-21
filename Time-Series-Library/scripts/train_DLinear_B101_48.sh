#!/bin/bash

# Parametri chiave per SegRNN:
  # seq_len (lunghezza della sequenza di input)
  # pred_len (lunghezza della previsione)
  # enc_in (numero di feature in input)
  # d_model (dimensione dello spazio latente della GRU)
  # dropout e learning_rate

# Imposta la GPU da utilizzare (se hai più GPU)
export CUDA_VISIBLE_DEVICES=0

# Percorso radice del dataset
ROOT_PATH=../data/final/

# Parametri comuni
DATA_FILE="B101_50khz_downsampled_reduced.csv"         # Usa il file di training (assicurati di avere anche validazione e test se necessario)
FEATURES="M"                  # Multivariato
SEQ_LEN=48                    # Lunghezza della sequenza in input
LABEL_LEN=24                  # Lunghezza dell'input per le etichette (separato dal forecasting horizon)
PRED_LEN=48                   # Lunghezza della previsione

DES="forecasting_48"

# Definisci il percorso all'eseguibile Python del virtual environment
PYTHON_EXE=../../.venv/Scripts/python.exe

# Definisci il percorso dello script run.py (senza spazi attorno all'uguale)
RUN_PY_PATH=../run.py

# Nome del modello
MODEL_NAME="DLinear"

echo "========================================"
echo "Avvio training per il modello: ${MODEL_NAME} con finestra${SEQ_LEN}"
echo "========================================"

$PYTHON_EXE -u $RUN_PY_PATH \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ${ROOT_PATH} \
  --data_path ${DATA_FILE} \
  --model_id B101_${SEQ_LEN}_${PRED_LEN} \
  --model ${MODEL_NAME} \
  --data custom \
  --features ${FEATURES} \
  --seq_len ${SEQ_LEN} \
  --label_len ${LABEL_LEN} \
  --pred_len ${PRED_LEN} \
  --target "Channel35" \
  --e_layers 7 \
  --d_layers 4 \
  --moving_avg 25 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --factor 10 \
  --enc_in 62 \
  --dec_in 62 \
  --batch_size 64 \
  --c_out 62 \
  --num_workers 0 \
  --des "${DES}" \
  --itr 1

echo "Training completato per ${MODEL_NAME}."