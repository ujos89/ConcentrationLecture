# ConcentrationLecture
## Overview
![Overview](./images/overview.jpeg)
figure 1. Our project overview

### Contributors
* [WoodoLee](https://github.com/ku-cylee)
* [lawkelvin33](https://github.com/lawkelvin33)
* [noparkee](https://github.com/noparkee)
* [ujos89](https://github.com/ujos89)

### Step0. Environment dependency


### Step1. video data to pickle (video2pickle.py)
```sh
$ python3 video2pickle.py --video [video_name] --savefile [file_name_to_save]
```
We used [ildoonet/tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation.git) to extract each body part informations

### Step2. data preprocessing (preprocessing.py)
```sh
$ python3 preprocessing.py --rawroot [raw_file_name] --label [contribute_or_not] --name [prepared_data_file_name]
```
매 프레임마다 L2(norm)을 사용해서 12개의 각 파트(상체)와 코 사이의 거리를 구한다.
결측 데이터를 각 항목의 중간값으로 채우고 정규화를 적용한다.
100개의 프레임 마다 각 파트의 분산을 구하고 라벨을 추가한다.
하나의 행이 하나의 데이터가 된다.

### Step3. Merge prepared dataset & shuffle

### Step4. Traing DNN

### Step5. Analysis
compared between size of data
