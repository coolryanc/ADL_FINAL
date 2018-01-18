# ADL_FINAL

## Introduction 簡介

* Team Name: 傻 B 好可愛 (或「傻逼好可愛」)
* Team Members:
  - R05922103 吳育騫
  - R05922105 陳俞安
  - R04922170 林于智

我們的 project 是使用 deep learning 的方式做到 real-time 辨別使用者的眼部表情與動作。
如此一來可以增加頭戴式顯示器 (特別是擴增實境類型) 的互動和輸入方式，甚至在雙手被限制/無法使用的狀況下更顯重要。

> 助教的 README 規定:
> 1. deadline: 1/18 23:59 逾期一天分數扣25%（超過4天即0分）
> 2. GitHub repository 裡面的 README.md 須包含 (1) 如何跑 training (2)如何跑 testing (3) 實驗環境描述（所需資料、系統、所需所有套件版本等）以上任何一項不完整都是扣1分
> 3. GitHub repository 包含可以運行的程式碼 （無法運行即0分）

## Environment 實驗環境描述

### Required Packages 所需套件

1. `Keras (2.1.2)`
2. `scikit-learn (0.19.1)`
3. `scikit-image (0.13.1)`
4. `scipy (1.0.0)`
5. `sklearn (0.0)`
6. `tensorflow (1.4.1)`
7. `Pillow (4.3.0)`
8. `pandas (0.22.0)`
9. `matplotlib (2.1.1)`
10. `h5py (2.7.1)`
11. `numpy (1.13.3)`
12. `opencv-python (3.3.0.10)`

> These are extracted from: `$pip list`.

## Data 所需資料

1. 請至 [TODO: link](https://www.google.com.tw) 下載並解壓縮
2. 將 `data/` 資料夾放至本 repo 最外層, i.e., 與這個 `README.md` 同層

> :warning: 原本 repo 裡面的 `data/` 可以直接覆蓋掉沒有關係的！

## How to train / test 如何訓練與測試

我們的所有 data 都是用我們自己的 binocular eye-tracker (from [Pupil Labs](https://pupil-labs.com/)) 搜集的。

我們總共找來 12 個人，因此有做兩種 training:
1. 針對每個人個別 train 一個 Personal Model
2. 用全部的 data 混在一起 train 一個 General Model

因此以下針對這兩種分別描述 training / testing 的方式

### Personal Model 個人模型

```sh
# please use python3 if your default python is of version 2
python PersonalModel.py <person>
```

執行時需要給的 `<person>` 參數應為那位 personal data 的所在資料夾。
比如說 `python PersonalModel.py 12_Te_Yan_Wu` 會為 Te-Yan Wu 先生 (我們其中一個好心的受試者) train 他個人的 model。

> :warning::warning::warning: 注意: 雖然 personal data 都在 `data/<their_name>/` 裡面，
> 但是只需要給裡面那層的資料夾名稱 (i.e., `<their name>`)，
> 而不需要連前面的 `data/` 都給

我們總共有 12 個人的 data, 他們的列表在此:
1. `1_Sandra`
2. `2_Giddy`
3. `3_Tey_Yin_Cheng`
4. `4_Kate`
5. `5_Sophia`
6. `6_Hugo`
7. `7_Ting_Chin_Xien`
8. `8_David`
9. `9_Jason`
10. `10_Low_Yan_Yu`
11. `11_Foong_Zhi_Sheng`
12. `12_Te_Yan_Wu`

### General Model 一般模型

```sh
# please use python3 if your default python is of version 2
python GeneralModel.py
```


## Model Structure 模型架構

The structure of our models is as follows:

```python
# Firstly, a convolutional layer:
Conv2D
MaxPooling2D
Dropout

# And then a second convolutional layer:
Conv2D
MaxPooling2D
Dropout

# flatten
Flatten

# fully-connected
Dense
Dropout

# two more fully-connected layers
Dense
Dense
```
