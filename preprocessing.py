import argparse
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math

body = ["Nos", "Nec", "Rsh", "Rel", "Rwr", "Lsh", "Lel", "Lwr",
       "Rey", "Ley", "Rea", "Lea"]

dist_dict = {}
var_dict = {}

w = 432     # 이거 나중에 처리해줘야해
h = 368

parser = argparse.ArgumentParser(description='for preprocessing tfpose data...')
parser.add_argument('--rawroot', type=str, required=True, help='raw data 경로 및 이름')
parser.add_argument('--label', type=int, required=True, help='concetrate or not')
parser.add_argument('--varname', type=str, required=True, help='variance data 저장할 경로 및 이름')
args = parser.parse_args()

'''file_root = input("파일 경로 (파일 이름 포함): ")
data_name = input
label = input("label: ")'''

### --- 함수 정의 --- ###
### 거리 계산하는 함수
def cal_dis(df, nose, other):       # str input으로
    h = []
    n_x = nose + '_X'
    n_y = nose + '_Y'
    o_x = other + '_X'
    o_y = other + '_Y'
    o_s = other + 'Score'

    if df[o_x] == 0 and df[o_y] and df[o_s] == 0:
        h.append(-1)
    else:
        c = math.sqrt((df[n_x] - df[o_x])**2 + (df[n_y] - df[o_y])**2 )
        h.append(c)
    
    return h

### 분산 계산하는 함수
def cal_var(df, num):
    i = 0
    var_lst = []
    while i <= len(df):
        if i + num <= len(df):
            temp_lst = df[i:i+num]
        else:
            temp_lst = df[i:]
        
        var = temp_lst.var()
        var_lst.append(var)

        i = i + num

    return var_lst
    
### --- ###

### 데이터 읽어오기
df = pd.read_pickle(args.rawroot)    

### w, h 곱해주기
for i in body:
    df[i + '_X'] = df[i + '_X'] * w
    df[i + '_Y'] = df[i + '_Y'] * h

### 거리 구해서 dictionary로 저장
for i in body:
    dist_dict[i] = cal_dis(df, 'Nos', i)    

### dictionary 자료형을 data frame 자료형으로
dist_frame = pd.DataFrame(dist_dict)
### Nos column drop하기. (자기 자신과의 거리니까)
dist_frame.drop('Nos', inplace=True)

# variance 계산해서 dictionary로 저장
for i in (dist_frame.columns):
    var_dict[i] = cal_var(dist_frame[i], 100)

# dictionary를 data frame으로
var_df = pd.DataFrame(var_dict) 

var_df['label'] = args.label

var_df.to_pickle("./result_data/var_" + args.varname + "_" + str(args.label) + ".pkl")
