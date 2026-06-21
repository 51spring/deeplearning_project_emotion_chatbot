@echo off
chcp 65001 >nul

set ROOT=emotion_chatbot

mkdir %ROOT%\data\raw
mkdir %ROOT%\data\processed
mkdir %ROOT%\data\nli
mkdir %ROOT%\data\preprocess
mkdir %ROOT%\models\roberta\checkpoints
mkdir %ROOT%\models\qwen\checkpoints
mkdir %ROOT%\pipeline
mkdir %ROOT%\backend\db
mkdir %ROOT%\frontend
mkdir %ROOT%\eval

:: data/processed 파일
type nul > %ROOT%\data\processed\emotion_train.csv
type nul > %ROOT%\data\processed\emotion_val.csv
type nul > %ROOT%\data\processed\emotion_calib.csv
type nul > %ROOT%\data\processed\nli_pairs.csv
type nul > %ROOT%\data\processed\qwen_finetune.jsonl
type nul > %ROOT%\data\processed\cbt_anchors.json

:: data/nli 파일
type nul > %ROOT%\data\nli\nli_pairs.csv

:: data/preprocess 파일
type nul > %ROOT%\data\preprocess\preprocess_emotion.py
type nul > %ROOT%\data\preprocess\preprocess_qwen.py
type nul > %ROOT%\data\preprocess\preprocess_aihub.py
type nul > %ROOT%\data\preprocess\build_nli_pairs.py

:: models/roberta 파일
type nul > %ROOT%\models\roberta\train_roberta.py
type nul > %ROOT%\models\roberta\inference_roberta.py
type nul > %ROOT%\models\roberta\temperature_scaling.py

:: models/qwen 파일
type nul > %ROOT%\models\qwen\train_qwen_lora.py
type nul > %ROOT%\models\qwen\inference_qwen.py

:: pipeline 파일
type nul > %ROOT%\pipeline\roberta_score.py
type nul > %ROOT%\pipeline\cbt_similarity.py
type nul > %ROOT%\pipeline\ensemble.py
type nul > %ROOT%\pipeline\ewma.py
type nul > %ROOT%\pipeline\wellness_score.py
type nul > %ROOT%\pipeline\score_pipeline.py

:: backend 파일
type nul > %ROOT%\backend\main.py
type nul > %ROOT%\backend\scheduler.py
type nul > %ROOT%\backend\daily_summary.py
type nul > %ROOT%\backend\crisis_handler.py
type nul > %ROOT%\backend\db\models.py
type nul > %ROOT%\backend\db\crud.py

:: eval 파일
type nul > %ROOT%\eval\eval_roberta.py
type nul > %ROOT%\eval\eval_crisis.py
type nul > %ROOT%\eval\eval_cbt.py
type nul > %ROOT%\eval\eval_qwen.py

echo.
echo 프로젝트 폴더 생성 완료: %ROOT%
pause
