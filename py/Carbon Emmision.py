import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('Carbon Emission.csv')
df = df.dropna()
df['CarbonEmission'] = pd.to_numeric(df['CarbonEmission'], errors='coerce')

median_emisi = df['CarbonEmission'].median()
df['Status_Emisi'] = df['CarbonEmission'].apply(lambda x: 1 if x > median_emisi else 0)

df['How Often Shower'] = df['How Often Shower'].astype('category')
kategori_shower = df['How Often Shower'].cat.categories.tolist()
df['How Often Shower'] = df['How Often Shower'].cat.codes

X = df[['How Often Shower', 'Vehicle Monthly Distance Km', 'How Many New Clothes Monthly', 'How Long Internet Daily Hour']]
Y = df['Status_Emisi']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model_dt = DecisionTreeClassifier(max_depth=3, random_state=42)
model_dt.fit(X_train, Y_train)

print("\n" + "="*50)
print("   SISTEM AI PREDIKSI STATUS JEJAK KARBON INDIVIDU   ")
print("="*50)
print("Silakan pilih opsi frekuensi mandi kamu:")
for i, opsi in enumerate(kategori_shower):
    print(f" [{i}] {opsi}")

try:
    input_shower = int(input("\nMasukkan nomor opsi mandi (misal: 0 atau 1): "))
    input_jarak = float(input("Masukkan jarak tempuh kendaraan per bulan (km): "))
    input_baju = int(input("Masukkan jumlah baju baru yang dibeli per bulan (pcs): "))
    input_internet = float(input("Masukkan durasi internetan per hari (jam): "))

    data_user = pd.DataFrame([[input_shower, input_jarak, input_baju, input_internet]], columns=X.columns)

    hasil_prediksi = model_dt.predict(data_user)

    print("\n" + "="*50)
    print("HASIL PREDIKSI STATUS EMISI ANDA:")
    if hasil_prediksi[0] == 1:
        print(" >> TINGGI EMISI (Berbahaya bagi Lingkungan) <<")
        print(" Rekomendasi: Kurangi mobilitas kendaraan pribadi & batasi fast fashion.")
    else:
        print(" >> RENDAH EMISI (Bagus dan Ramah Lingkungan) <<")
        print(" Pertahankan gaya hidup hemat energi dan konsumsi bertanggung jawab!")
    print("="*50 + "\n")

except ValueError:
    print("\n[Error] Harap masukkan data dalam bentuk angka yang valid!\n")
