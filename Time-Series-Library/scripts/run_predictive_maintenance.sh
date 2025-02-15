#!/bin/bash

# Imposta la GPU da utilizzare (se hai più GPU)
export CUDA_VISIBLE_DEVICES=0

# Percorso radice del dataset
ROOT_PATH=../data/final/

# Parametri comuni
DATA_FILE="cleaned_predictive_maintenance.csv"         # Usa il file di training (assicurati di avere anche validazione e test se necessario)
FEATURES="M"                  # Multivariato
SEQ_LEN=24                    # Lunghezza della sequenza in input
LABEL_LEN=24                  # Lunghezza dell'input per le etichette (separato dal forecasting horizon)
PRED_LEN=24                   # Lunghezza della previsione
E_LAYERS=2                    # Numero di layer per l'encoder
D_LAYERS=1                    # Numero di layer per il decoder
ENC_IN=6                      # Numero di feature in input (modifica in base al tuo dataset)
DEC_IN=6                      # Numero di feature in input per il decoder
C_OUT=6                       # Numero di feature in output
DES="forecasting_24"

# Definisci il percorso all'eseguibile Python del virtual environment
PYTHON_EXE=../../.venv/Scripts/python.exe

# Definisci il percorso dello script run.py (senza spazi attorno all'uguale)
RUN_PY_PATH=../run.py

# Ciclo sui modelli da testare: TSMixer e TimesNet
for model_name in TSMixer
do
    echo "========================================"
    echo "Avvio training per il modello: ${model_name}"
    echo "========================================"

    $PYTHON_EXE -u $RUN_PY_PATH \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ${ROOT_PATH} \
      --data_path ${DATA_FILE} \
      --model_id ${model_name}_24_forecast \
      --model ${model_name} \
      --data custom \
      --features ${FEATURES} \
      --target "Process temperature [K]" \
      --seq_len ${SEQ_LEN} \
      --label_len ${LABEL_LEN} \
      --pred_len ${PRED_LEN} \
      --e_layers ${E_LAYERS} \
      --d_layers ${D_LAYERS} \
      --enc_in ${ENC_IN} \
      --dec_in ${DEC_IN} \
      --c_out ${C_OUT} \
      --des "${DES} usando ${model_name}" \
      --itr 1

    echo "Training completato per ${model_name}."
    echo ""
done
