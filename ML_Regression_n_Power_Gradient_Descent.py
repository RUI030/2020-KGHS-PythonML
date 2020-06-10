import numpy as np
import matplotlib.pyplot as plt

#回歸函數(數字版)
def regression_function_num(x, beta):
  ans = 0
  for i in range(len(beta)):
    ans += beta[i] * (x**i)
  return ans
#回歸函數(陣列版)
def regression_function(arr_x, beta):
  LIST = []
  for i in range(len(arr_x)):
    ans = 0
    for j in range(len(beta)):
      ans += beta[j] * (arr_x[i]**j)
    LIST.append(ans)
  return np.array(LIST)
#損失函數
def loss_function(arr_x, arr_y, beta):
  e_2 = arr_y - regression_function(arr_x, beta)
  e_2 =e_2**2
  return 0.5 * (np.sum(e_2))
#將回歸函數係數偏微分的函式
def par_beta_function(arr_x, arr_y, beta):
  par_beta = []
  num = 0
  for i in range(len(beta)):
    num = - np.dot(arr_y - regression_function(arr_x, beta) , (arr_x**i))
    par_beta.append(num)
  return np.array(par_beta)
#梯度下降
def gradient_descent(beta, eta, slope):
  return beta - eta*slope
#標準化數據的函數
def standardize(data):
  return (data - data.mean()) / data.std()
#歸一化數據的函數
def normalize(data):
  return (data - data.min()) / (data.max() - data.min())

#輸入原始資料
list_x = [8,-10,13,-16,23,-30,-34,45,-50,53,-57,62,-66,69,-75,81,-88,92,-2,5]
list_y = [5.7,8.8,18,21,30,42,80,130,200,263,248,327,367,433,486,515,756,818,0.6,1]

#將原始資料轉成陣列並分割為訓練資料與測試資料
data_x_raw = np.array(list_x)
data_y_raw = np.array(list_y)
percentage =0.8
divide = int(len(data_x_raw)*percentage)
train_x_raw = data_x_raw[:divide]
train_y_raw = data_y_raw[:divide]
check_x_raw = data_x_raw[divide:]
check_y_raw = data_y_raw[divide:]
#原始資料資料標準化版本
data_x_std = standardize(data_x_raw)
data_y_std = standardize(data_y_raw)
train_x_std = data_x_std[divide:]
train_y_std = data_y_std[divide:]
check_x_std = data_x_std[:divide]
check_y_std = data_y_std[:divide]
#歸一化版本
data_x_norm = normalize(data_x_raw)
data_y_norm = normalize(data_y_raw)
train_x_norm = data_x_norm[:divide]
train_y_norm = data_y_norm[:divide]
check_x_norm = data_x_norm[divide:]
check_y_norm = data_y_norm[divide:]

#設定
power = 3    #回歸函數次方
times = 10000    #訓練次數

#初始化
beta = np.random.random(size = (power + 1))    #回歸函數係數的隨機陣列
eta = np.full(len(beta),(10**-2))
#建立一個存loss function值的空列表
loss_record = []

#選用資料
arr_x = train_x_norm    #這裡選用歸一化的資料處理，可更改
arr_y = train_y_raw
test_x = check_x_norm
test_y = check_y_raw
#開始訓練
for i in range(times):
  par_beta = par_beta_function(arr_x, arr_y, beta)
  beta = gradient_descent(beta, eta, par_beta)
  #紀錄每次訓練後損失函數的值
  loss_record.append(loss_function(arr_x, arr_y, beta))
  
#設定一下matplotlib輸出時的標題、顏色等等
plt.figure(figsize = (8,8), dpi = 100)
plt.title("Regression Equation", fontsize=20, color="#888888") 
plt.xlabel("x axis", fontsize=14, color="#cccccc") 
plt.ylabel("y axis", fontsize=14, color="#cccccc")
plt.tick_params(axis='x', colors='#cccccc')
plt.tick_params(axis='y', colors='#cccccc') 
#使用matplotlib畫出回歸函式
x = np.linspace(0,1,10000)
y = regression_function(x, beta)
plt.plot(x,y, color='STEELBLUE')
#使用matplotlib畫出訓練資料與測試資料
plt.plot(arr_x,arr_y, "o", color="ORANGE")
plt.plot(test_x,test_y, "o", color="teal")
plt.show()    #將剛剛畫上的資料輸出成一張圖顯示

#輸出(顯示)回歸函式(文字公式)
print('regression equation :  y = ',sep='',end='')
for i in range(len(beta)):
  if i==1:
    if beta[i]>0:
      print(' + ', beta[i],'x',sep='',end='')
    elif beta[i]<0:
      print(' - ', -beta[i],'x',sep='',end='')
  elif i:
    if beta[i]>0:
      print(' + ', beta[i],'x^',i,sep='',end='')
    elif beta[i]<0:
      print(' - ', -beta[i],'x^',i,sep='',end='')
  else:
    print(beta[i],sep='',end='')
#使用matplotlib畫出Loss function值隨訓練次數變化的情形
dot_num = 50  #要描的點的數量，不能超過loss record長度
plt.figure(figsize=(8,8), dpi=100)
plt.title("Loss Function", fontsize=20, color="#888888") 
plt.xlabel("Training Times", fontsize=14, color="#cccccc") 
plt.ylabel("Loss", fontsize=14, color="#cccccc")
plt.tick_params(axis='x', colors='#cccccc')
plt.tick_params(axis='y', colors='#cccccc') 
for i in range(0, len(loss_record), int(len(loss_record) / dot_num)):
  plt.plot(i,loss_record[i], '.', color='MEDIUMVIOLETRED')
plt.show()
