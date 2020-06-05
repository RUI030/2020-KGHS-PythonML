import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(4,3),dpi=200)
plt.title("Linear Regression",fontsize=20,color="#888888")
plt.tick_params(axis='x',colors='#cccccc')
plt.tick_params(axis='y',colors='#cccccc')
plt.xlabel("X axis",color="#cccccc")
plt.ylabel("Y axis",color="#cccccc")

def Sum(LIST):
  Len=len(LIST)
  SUM=0
  for i in range(Len):
    SUM+=LIST[i]
  return SUM

def Sum_Var_Pow_N(LIST,N):
  Len=len(LIST)
  Pow=N
  SUM=0
  for i in range(Len):
    SUM+=LIST[i]**Pow
  return SUM

def Sum_XY(LIST_X,LIST_Y):
  Len=len(LIST_X)
  SUM=0
  for i in range(Len):
    SUM+=(LIST_X[i]*LIST_Y[i])
  return SUM

def Draw_Point(X,Y):
  plt.plot(X,Y,"o",color="Teal")

def Draw_Line(A,B,START_X=-3,END_X=10,POINT=10000):
  LIST=[]
  X=np.arange(START_X,END_X)
  Y=A*X+B
  plt.plot(X,Y,color="Mediumvioletred")
  return 0

X=[]
Y=[]

DataCount = int(input("input data sets number:"))
for i in range(DataCount):
  INPUT=float(input("X:"))
  X.append(INPUT)
  INPUT=float(input("Y:"))
  Y.append(INPUT)

Sigma_X=Sum(X)
Sigma_Y=Sum(Y)
Sigma_X_2=Sum_Var_Pow_N(X,2)
Sigma_Y_2=Sum_Var_Pow_N(Y,2)
Sigma_XY=Sum_XY(X,Y)

W1=(Sigma_X*Sigma_Y-DataCount*Sigma_XY)/(Sigma_X**2-DataCount*Sigma_X_2)
W2=((Sigma_X*Sigma_XY)-(Sigma_X_2*Sigma_Y))/(Sigma_X**2-DataCount*Sigma_X_2)

START=int(input('START_X:'))
END=int(input('STOP_X:'))
POINT=int(input('POINT_NUMBER:'))
Draw_Line(W1,W2,START,END,POINT)
for i in range(DataCount):
  Draw_Point(X[i],Y[i])

print("Regression Line: Y =",W1,"X ",end="")
if W2>0:
  print("+",W2)
elif W2<0:
  print("-",-W2)
else:
  print("")

plt.show()
