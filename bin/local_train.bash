TRAIN_DATA=$(pwd)/data/training_data.csv
EVAL_DATA=$(pwd)/data/validation_data.csv
MODEL_DIR=/tmp/proteinrnn
rm -rf $MODEL_DIR
gcloud ml-engine local train --module-name trainer.task --package-path trainer/ -- --train-files $TRAIN_DATA --eval-files $EVAL_DATA --job-dir $MODEL_DIR
