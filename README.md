# ODM
Object Detection Metrics

## _I. Introduction_

A Code that helps to build graphs and calculate metrics such as: **Confusion matrix, Precision, Recall, AUC ROC**


## _II. Instructions_

### _II.I. Prepocessing data_

##### 1) Actual data

Actual data must be in the same format as in this [file](example/actual.csv)

| image_id | xmin | ymin | xmax | ymax | label |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0003.png | 278 | 902 | 738 | 1160 | dog |
| 0005.png | 94 | 591 | 1196 | 1052 | cat |
| 0005.png | 8 | 138 | 1179 | 620 | dog |
| 0005.png | 13 | 148 | 1173 | 589 | cat |
| 0008.png | 58 | 1224 | 1117 | 1519 | cat |
| 0008.png | 58 | 64 | 1100 | 761 | dog |

##### 2) Detected data

Detected data must be in the same format as in this [file](example/detected.csv)

| image_id | xmin | ymin | xmax | ymax | label | prob |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0003.png | 235 | 887 | 1088 | 1174 | dog | 1.0 |
| 0005.png | 38 | 128 | 1178 | 588 | cat | 1.0 |
| 0008.png | 43 | 1229 | 1141 | 1517 | cat | 0.98 |
| 0008.png | 68 | 104 | 1107 | 760 | dog | 1.0 |
| 0008.png | 68 | 104 | 1107 | 760 | dog | 1.0 |

### _II.II. Run script_
```
python -m venv venv
source venv/bin/activate
python example_main.py
```

## _III. Requirements_
```
python > 3.6
matplotlib > 3.3
numpy > 1.19
pandas > 1.1
scikit-learn > 0.23
```

