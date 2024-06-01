'''
Tugas Besar Permodelan dan Identifikasi Sistem

System Identification of a DC Motor Using ARX Model

Nama : Muhammad Husni Muttaqin

NIM : 23223303

Teknik Kendali dan Sistem Cerdas STEI ITB

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data_dummy.csv')


u = df.iloc[:, 0].to_numpy() 
y = df.iloc[:, 1].to_numpy()

print(u)
print(y)
# Tentukan order model ARX

n = 2  # order AR
m = 1  # order X

# Buat matriks regresi Phi
N = len(y)
Phi = np.zeros((N-n, n+m))

for i in range(n, N):
    Phi[i-n, :n] = -y[i-n:i][::-1]  # bagian AR
    Phi[i-n, n:] = u[i-m:i][::-1]   # bagian X

# Vektor output
Y = y[n:]

# Cetak matriks Phi dan Y untuk verifikasi
print("Matriks Phi:")
print(Phi)

print("Vektor Y:")
print(Y)

# Estimasi parameter
Theta = np.linalg.inv(np.dot(Phi.T,Phi)) @ np.dot(Phi.T, Y)

# Pisahkan parameter menjadi a dan b
print("Theta")
print(Theta)
a = Theta[:n]
b = Theta[n:]

print("Parameter a:", a)
print("Parameter b:", b)


###############################################################################

a_pred_val = 0
b_pred_val = 0


# Data input (u) dan output (y) yang telah digunakan sebelumnya
u = df.iloc[:, 0].to_numpy() 
y_actual = df.iloc[:, 1].to_numpy()

# Parameter yang telah diestimasi
a_measured=a
b_measured=b

# Tentukan order model ARX
n = len(a)  # order AR
m = len(b)  # order X

# Inisialisasi array untuk prediksi output
y_pred = np.zeros(len(y_actual))



# Hitung prediksi output menggunakan parameter yang diestimasi

print("validation")
print(a, b)

        
for t in range(n, len(y_actual)):
    #measure a value
    for a_counter in range(0,n):
        a_pred_val = (-a_measured[a_counter] * y_actual[t]) + a_pred_val
        
        
    
    #measure b value
    for b_counter in range(0,m):
        b_pred_val = (b_measured[b_counter] * y_actual[t-2]) + b_pred_val
        
    #measure y prediction
        
    y_pred[t] = a_pred_val + b_pred_val
    #y_pred[t] = -a_measured[0] * y_actual[t-1] - a_measured[1] * y_actual[t-2] + b_measured[0] * u[t-1] + b_measured[1] * u[t-2]
    
    #emptying buffer
    a_pred_val = 0
    b_pred_val = 0
    

    
        
# Cetak prediksi dan output asli untuk verifikasi
print("Output prediksi (y_pred):", y_pred[n:])
print("Output asli (y_actual):", y_actual[n:])

# Hitung error prediksi
error = y_actual[n:] - y_pred[n:]
print("Error prediksi:", error)

# Hitung Mean Squared Error (MSE) sebagai ukuran kinerja model
mse = np.mean(error**2)
print("Mean Squared Error (MSE):", mse)





# Membuat subplot (2 baris, 2 kolom)
plt.subplot(2, 2, 1)  # Subplot pertama di kiri atas
plt.plot(u, label='u', color='blue')  # Plot u
plt.plot(y, label='y', color='red')  # Plot u
plt.title('Plot of u')  # Judul subplot pertama
plt.legend()  # Menambahkan legenda

plt.subplot(2, 2, 2)  # Subplot kedua di kanan atas
plt.plot(y, label='y act', color='red')  # Plot y
plt.plot(y_pred, label='y pred', color='blue')  # Plot y
plt.title('Plot of y')  # Judul subplot kedua
plt.legend()  # Menambahkan legenda

plt.subplot(2, 2, 3)  # Subplot ketiga di kiri bawah
plt.plot(u, label='u', color='green')  # Plot u
plt.title('Plot of u')  # Judul subplot ketiga
plt.legend()  # Menambahkan legenda

plt.subplot(2, 2, 4)  # Subplot keempat di kanan bawah
plt.plot(y, label='y', color='orange')  # Plot y
plt.title('Plot of y')  # Judul subplot keempat
plt.legend()  # Menambahkan legenda

plt.suptitle('System Identification of a DC Motor Using ARX Model', fontsize=13, fontweight='bold')
plt.tight_layout()  # Mengatur tata letak subplot agar tidak tumpang tindih
plt.show()  # Menampilkan kedua subplot