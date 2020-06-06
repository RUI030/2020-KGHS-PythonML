import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as plt_font
from IPython import display

def regression_function_num(x, beta):
  ans = 0
  for i in range(len(beta)):
    ans += beta[i] * (x**i)
  return ans

def regression_function(arr_x, beta):
  LIST = []
  for i in range(len(arr_x)):
    ans = 0
    for j in range(len(beta)):
      ans += beta[j] * (arr_x[i]**j)
    LIST.append(ans)
  return np.array(LIST)

def loss_function(arr_x, arr_y, beta):
  e_2 = (arr_y - regression_function(arr_x, beta))**2
  return 0.5 * (np.sum(e_2))

def par_beta_function(arr_x, arr_y, beta):
  par_beta = []
  num = 0
  for i in range(len(beta)):
    num = - np.dot(arr_y - regression_function(arr_x, beta) , (arr_x**i))
    par_beta.append(num)
  return np.array(par_beta)

def gradient_descent(beta, eta, slope):
  return beta - eta*slope

def standardize(data):
  return (data - data.mean()) / data.std()

def normalize(data):
  return (data - data.min()) / (data.max() - data.min())

#original data
list_x = [8,-10,13,-16,23,-30,-34,45,-50,53,-57,62,-66,69,-75,81,-88,92,-2,5]
list_y = [5.7,8.8,18,21,30,42,80,130,200,263,248,327,367,433,486,515,756,818,0.6,1]
#type:list->array , divide training data & testing data
data_x = np.array(list_x)
data_y = np.array(list_y)
divide = int(len(data_x)*0.8)
train_x = data_x[:divide]
train_y = data_y[:divide]
check_x = data_x[divide:]
check_y = data_y[divide:]
#stdandardized data
data_x_std = standardize(data_x)
data_y_std = standardize(data_y)
train_x_std = data_x_std[divide:]
train_y_std = data_y_std[divide:]
check_x_std = data_x_std[:divide]
check_y_std = data_y_std[:divide]
#normalized data
data_x_norm = normalize(data_x)
data_y_norm = normalize(data_y)
train_x_norm = data_x_norm[:divide]
train_y_norm = data_y_norm[:divide]
check_x_norm = data_x_norm[divide:]
check_y_norm = data_y_norm[divide:]

#setting
power = 2
times = 20000
beta = np.random.random(size = (power + 1))
eta = np.full(len(beta),(10**-2))
#record initialize
loss_record = []

#regression
arr_x = train_x_norm
arr_y = train_y
for i in range(times):
  par_beta = par_beta_function(arr_x, arr_y, beta)
  beta = gradient_descent(beta, eta, par_beta)
  #record
  loss_record.append(loss_function(arr_x, arr_y, beta))
  
#draw
plt.figure(figsize = (8,8), dpi = 100)
plt.title("Regression Equation", fontsize=20, color="#888888") 
plt.xlabel("x axis", fontsize=14, color="#cccccc") 
plt.ylabel("y axis", fontsize=14, color="#cccccc")
plt.tick_params(axis='x', colors='#cccccc')
plt.tick_params(axis='y', colors='#cccccc') 
#draw regression equation
x = np.linspace(0,1,10000)
y = regression_function(x, beta)
plt.plot(x,y, color='STEELBLUE')
#draw data(dot)
plt.plot(arr_x,arr_y, "o", color="ORANGE")
plt.plot(check_x_norm,check_y, "o", color="teal")
plt.show()

#print regression equation
print('regression equation :  y = ',sep='',end='')
for i in range(len(beta)):
  if i==1:
    print(beta[i],'x',sep='',end='')
  elif i:
    print(beta[i],'x^',i,sep='',end='')
  else:
    print(beta[i],sep='',end='')
  if(len(beta)-i):
    print(' + ',sep='',end='')

#draw loss funtion
plt.figure(figsize=(8,8), dpi=100)
plt.title("Loss Function", fontsize=20, color="#888888") 
plt.xlabel("Training Times", fontsize=14, color="#cccccc") 
plt.ylabel("Loss", fontsize=14, color="#cccccc")
plt.tick_params(axis='x', colors='#cccccc')
plt.tick_params(axis='y', colors='#cccccc') 
for i in range(0, len(loss_record), int(len(loss_record) / 100)):
  plt.plot(loss_record, ':o', color='MEDIUMVIOLETRED')
plt.show()
