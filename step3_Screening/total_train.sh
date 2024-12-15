source ../ml_venv/bin/activate

python step1_preprocess.py
python step2_split_data.py

for fold_num in 0 1 2 3 4; do
    python step3_run.py --fold_num $fold_num 
done

echo "All done"