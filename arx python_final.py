'''
Tugas Besar Permodelan dan Identifikasi Sistem

System Identification of a DC Motor Using ARX Model

Nama : Muhammad Husni Muttaqin

NIM : 23223303

Teknik Kendali dan Sistem Cerdas STEI ITB

'''

print("==================== PROGRAM START ==============================")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('data_uji.csv')
dv = pd.read_csv('data_validasi.csv')


u = df.iloc[:, 0].to_numpy() 
y = df.iloc[:, 1].to_numpy()

u_validasi = dv.iloc[:, 0].to_numpy()
y_validasi = dv.iloc[:, 1].to_numpy() 


print("input matrices :")
print(u)
print("output matrices :")
print(y)


n = 2  # order AR
m = 0 # order X

N = len(y)
Phi = np.zeros((N-n, n+m))

############################################################

for i in range(n, N):
    Phi[i-n, :n] = -y[i-n:i][::-1]  
    Phi[i-n, n:] = u[i-m:i][::-1]   

Y = y[n:]


print("Phi matrices:")
print(Phi)

print("Y vectors:")
print(Y)


Theta = np.linalg.inv(np.dot(Phi.T,Phi)) @ np.dot(Phi.T, Y)


print("Theta matrices :")
print(Theta)
a = Theta[:n]
b = Theta[n:]

print("estimated a constanta:", a)
print("estimated b constanta:", b)


###############################################################################

a_pred_val = 0
b_pred_val = 0



u = df.iloc[:, 0].to_numpy() 
y_actual = df.iloc[:, 1].to_numpy()


a_measured=a
b_measured=b


n = len(a) 
m = len(b) 


y_pred = np.zeros(len(y_actual))



print("prediction")
print(a, b)

        
for t in range(n, len(y_actual)):

    
    for a_counter in range(0,n):
        a_pred_val = (-a_measured[a_counter] * y_actual[t-1]) + a_pred_val
        
    
    
    for b_counter in range(0,m):
        b_pred_val = (b_measured[b_counter] * u[t-2]) + b_pred_val
        
    
        
    y_pred[t] = a_pred_val + b_pred_val
    
    #emptying buffer
    a_pred_val = 0
    b_pred_val = 0
    

print("Output prediction (y_pred):", y_pred[n:])
print("Output actual (y_actual):", y_actual[n:])


error = y_actual[n:] - y_pred[n:]
print("prediction Error :", error)


mse = np.mean(error**2)
###############################################################

print("validation")

a_validation = 0
b_validation = 0


u_validation = dv.iloc[:, 0].to_numpy() 
y_validation = dv.iloc[:, 1].to_numpy()
print(len(y_validation))


y_validation_pred = np.zeros(len(y_validation))


for t in range(n, len(y_validation)):

    
    for a_counter in range(0,n):
        a_pred_val = (-a_measured[a_counter] * y_validation[t-1]) + a_pred_val
        
    
    
    for b_counter in range(0,m):
        b_pred_val = (b_measured[b_counter] * u_validation[t-2]) + b_pred_val
        
   
        
    y_validation_pred[t] = a_pred_val + b_pred_val
    
    
    a_pred_val = 0
    b_pred_val = 0


print(y_validation_pred)  
print(y_validation)   


print("Mean Squared Error (MSE):", mse)

error_validation = y_validation_pred[n:] - y_validation[n:]
print("validation Error :", error_validation)

mse_validation = np.mean(error_validation**2)
print("Mean Squared Error (MSE) validation:", mse_validation)

plt.subplot(2, 2, 1) 
plt.plot(y, label='output', color='red') 
plt.plot(u, label='input', color='#028391')  
plt.title('grafik input output') 
plt.legend()  

plt.subplot(2, 2, 3) 
plt.plot(y, label='y act', color='blue',linestyle='dashed')  
plt.plot(y_pred, label='y pred', color='#00ff00',linestyle='dashed') 
plt.title('grafik output prediksi dan aktual')  
plt.legend()

plt.subplot(2, 2, 4) 
plt.plot(error, label='e prediksi', color='orange') 
plt.plot(error_validation, label='e validasi', color='#007F73')  
plt.title('error estimasi')  
plt.legend() 

plt.subplot(2, 2, 2)  
plt.plot(y_validation, label='y actual', color='maroon')  
plt.plot(y_validation_pred, label='y pred', color='#3AA6B9',linestyle='dashed')  
plt.title('grafik validasi')  
plt.legend()  

fig = plt.gcf()
fig.text(0.5, 0.01, str("a = ") + str(a)+ str(" b = ") + str(b) + str("\n MSE = ") + str(mse) + str(" MSE Validation: ") + str(mse_validation), ha='center')
#


plt.suptitle('System Identification of a DC Motor Using ARX Model \n Muhammad Husni Muttaqin (23223303)', fontsize=13, fontweight='bold')
plt.tight_layout()

plt.show() 

