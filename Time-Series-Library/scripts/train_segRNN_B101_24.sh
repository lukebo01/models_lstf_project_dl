#!/bin/bash

# Parametri chiave per SegRNN:
  # seq_len (lunghezza della sequenza di input)
  # pred_len (lunghezza della previsione)
  # seg_len (lunghezza di ciascun segmento)
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
SEQ_LEN=24                    # Lunghezza della sequenza in input
LABEL_LEN=24                  # Lunghezza dell'input per le etichette (separato dal forecasting horizon)
PRED_LEN=24                   # Lunghezza della previsione
SEG_LEN=6                   # Lunghezza di ogni segmento
DES="forecasting_24"

# Definisci il percorso all'eseguibile Python del virtual environment
PYTHON_EXE=../../.venv/Scripts/python.exe

# Definisci il percorso dello script run.py (senza spazi attorno all'uguale)
RUN_PY_PATH=../run.py

# Nome del modello
MODEL_NAME="SegRNN"

echo "========================================"
echo "Avvio training per il modello: ${MODEL_NAME} con finestra${SEQ_LEN}"
echo "========================================"

$PYTHON_EXE -u $RUN_PY_PATH \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ${ROOT_PATH} \
  --data_path ${DATA_FILE} \
  --model_id ETTh1_${SEQ_LEN}_${PRED_LEN} \
  --model ${MODEL_NAME} \
  --data custom \
  --features ${FEATURES} \
  --seq_len ${SEQ_LEN} \
  --label_len ${LABEL_LEN} \
  --pred_len ${PRED_LEN} \
  --seg_len ${SEG_LEN} \
  --target "Channel35" \
  --enc_in 62 \
  --d_model 512 \
  --dropout 0.5 \
  --learning_rate 0.0001 \
  --des "${DES} usando ${MODEL_NAME}" \
  --itr 1

echo "Training completato per ${MODEL_NAME}."

#--task_name long_term_forecast         # Specifica il task: forecasting a lungo termine
#--is_training 1                      # Attiva la modalità training (1 = training)
#--root_path ${ROOT_PATH}             # Percorso radice del dataset
#--data_path ${DATA_FILE}             # Nome del file CSV contenente i dati
#--model_id ETTh1_${SEQ_LEN}_${PRED_LEN}  # ID del modello, costruito usando SEQ_LEN e PRED_LEN per identificarlo
#--model ${MODEL_NAME}                # Nome del modello da utilizzare (in questo caso, SegRNN)
#--data custom                        # Tipo di dataset (custom)
#--features ${FEATURES}               # Tipo di features: "M" indica multivariato
#--seq_len ${SEQ_LEN}                 # Lunghezza totale della sequenza di input
#--label_len ${LABEL_LEN}             # Lunghezza utilizzata per le etichette
#--pred_len ${PRED_LEN}               # Lunghezza della finestra di previsione (output)
#--seg_len $((SEQ_LEN / 4))             # Lunghezza di ogni segmento (1/4 di SEQ_LEN; es. 24/4 = 6)
#--enc_in 7                           # Numero di feature in input per l'encoder (canali)
#--d_model 512                        # Dimensione del modello (dimensione dello spazio latente)
#--dropout 0.5                        # Tasso di dropout per la regolarizzazione
#--learning_rate 0.0001               # Tasso di apprendimento per l'ottimizzatore
#--des "${DES} usando ${MODEL_NAME}"  # Descrizione dell'esperimento (include il modello usato)
#--itr 1                              # Numero di iterazioni/repliche dell'esperimento
