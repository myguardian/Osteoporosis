#! /bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

python3 "$SCRIPTPATH/LoadModel.py" "random_forest_model.sav" "Clean_Data_Main.csv"
python3 "$SCRIPTPATH/RFC.py" "Clean_Data_Main.csv"
python3 "$SCRIPTPATH/Classifier.py" "Clean_Data_Main.csv"
python3 "$SCRIPTPATH/Simplified_Classifier.py" "Clean_Data_Main.csv"
