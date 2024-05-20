import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
import calendar
import re
import io
import base64

# Fungsi untuk membaca dan memproses data
def process_data():
    file_path = 'mean.csv'
    data = pd.read_csv(file_path, sep=';')
    negara_komoditas = data.groupby(['Negara', 'Komoditas']).size().reset_index(name='count')
    negara_komoditas_grouped = negara_komoditas.groupby('Negara')['Komoditas'].apply(', '.join).reset_index()
    return negara_komoditas_grouped

# Fungsi prediksi harga
def linear_regression(group):
    model = LinearRegression()
    X = group['Volume'].values.reshape(-1, 1)
    y = group['Tabel Harga']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    return model, scaler

def predict_price(model, additional_data, min_price, scaler):
    additional_data_scaled = scaler.transform([[additional_data]])
    predicted_price = model.predict(additional_data_scaled)
    return max(round(predicted_price[0], 2), min_price)

# Fungsi utama untuk membuat grafik
def generate_graph(komoditas, negara, data_path):
    data = pd.read_csv(data_path, sep=';')
    regression_models = data.groupby(['Komoditas', 'Negara']).apply(lambda x: linear_regression(x)).to_dict()

    MIN_PREDICTED_PRICE = 0.01
    for key, (model, scaler) in regression_models.items():
        if key[0].lower() == komoditas and key[1].lower() == negara:
            additional_data = 0
            predicted_prices = []
            current_year, current_month = 2024, 1
            months = []
            prices = []

            for _ in range(36):
                predicted_price = predict_price(model, additional_data, MIN_PREDICTED_PRICE, scaler)
                predicted_prices.append(predicted_price)
                additional_data = predicted_price
                months.append(f"{calendar.month_name[current_month]} {current_year}")
                prices.append(predicted_price)
                current_month += 1
                if current_month > 12:
                    current_month = 1
                    current_year += 1

            plt.figure(figsize=(10, 6))
            plt.plot(months, prices, marker='o')
            plt.xticks(rotation=45, ha='right')
            plt.xlabel('Bulan Tahun')
            plt.ylabel('Harga Prediksi')
            plt.title(f'Prediksi Harga {komoditas.title()} di {negara.title()}')
            plt.tight_layout()
            plt.grid(True)
            st.pyplot(plt)
            plt.close()
            return True
    return False

# Fungsi utama untuk membuat grafik regresi linear
def linear_regression_plot(komoditas, negara):
    file_path = 'mean.csv'
    data = pd.read_csv(file_path, sep=';')
    data.dropna(subset=['Volume', 'Tabel Harga'], inplace=True)
    komoditas = re.sub(r'\s+', ' ', komoditas.strip())
    negara = re.sub(r'\s+', ' ', negara.strip())

    filtered_data = data[(data['Komoditas'].str.upper() == komoditas.upper()) & (data['Negara'].str.upper() == negara.upper())]

    if not filtered_data.empty:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(filtered_data[['Volume']])
        y = filtered_data['Tabel Harga']

        model = make_pipeline(LinearRegression())
        model.fit(X_scaled, y)

        beta_0 = model.named_steps['linearregression'].intercept_
        beta_1 = model.named_steps['linearregression'].coef_[0]

        equation = f"Y = {beta_0:.2f} + {beta_1:.2f}X"

        plt.figure(figsize=(10, 6))
        plt.scatter(X_scaled, y, color='blue', label='Data')
        plt.plot(X_scaled, model.predict(X_scaled), color='red', label='Regresi Linear')
        plt.title(f'Regresi Linear: Volume vs Harga ({komoditas} di {negara})')
        plt.xlabel('Volume (Scaled)')
        plt.ylabel('Harga')
        plt.legend()
        plt.grid(True)

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return equation, graph_url
    else:
        return None, None

def main():
    st.set_page_config(page_title="Aplikasi Prediksi Harga Komoditas", layout="wide")
    menu = option_menu(
        menu_title=None,  # Judul menu, jika None tidak ada judul
        options=["Home", "Histori Data", "Prediksi", "Grafik Harga", "Grafik Linear"],  # Opsi menu
        icons=["house", "clock", "graph-up-arrow", "bar-chart-line", "calculator"],  # Ikon untuk setiap opsi
        menu_icon="cast",  # Ikon untuk menu secara keseluruhan
        default_index=0,  # Indeks opsi yang dipilih secara default
        orientation="horizontal"  # Orientasi menu, bisa "horizontal" atau "vertical"
    )

    if menu == "Home":
        st.markdown(
            """
            <h1 style='text-align: center; margin-top: 50px;'>Selamat datang di platform kami!</h1>
            <p style='text-align: center; margin-top: 20px;'>Temukan prediksi terkini untuk nilai harga dan volume produksi dalam industri perkebunan <br>untuk membantu Anda mengambil keputusan yang lebih cerdas dan tepat waktu.</p>
            """,
            unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 4, 1])  # Membagi layout menjadi 3 bagian
        with col2:
            st.image('sawi.jpg' , use_column_width=True, output_format='JPEG')
        col1, col2, col3 , col4= st.columns([1, 1, 1, 1])  # Membagi layout menjadi 3 bagian
     
        with col2:
            st.markdown(
                """
                <h2 class="text-lg font-semibold m-4">Komoditas Semusim</h2>
                <p class="mb-2">&#10004; Adas <br>
                &#10004; Manis <br>
                &#10004; Asam <br>
                &#10004; Biji Wijen <br>
                &#10004; Cengkeh <br>
                &#10004; Gingseng <br>
                &#10004; Jintan <br>
                &#10004; Kakao <br>
                &#10004; Karet <br>
                &#10004; Kelapa <br>
                &#10004; Kopi <br>
                &#10004; Lada <br>
                &#10004; Pala <br>
                &#10004; Panili <br>
                &#10004; Pinang <br>
                &#10004; Tembakau </p>
                """,
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                """
                <h2 class="text-lg font-semibold m-4">Komoditas Tahunan</h2>
                <p class="mb-2">&#10004; Benih Tanaman Perkebunan <br>
                &#10004; Kacang Makademia <br>
                &#10004; Kacang Mede <br>
                &#10004; Kapas <br>
                &#10004; Kapuk <br>
                &#10004; Kayu Manis <br>
                &#10004; Kelapa Sawit <br>
                &#10004; Kemiri <br>
                &#10004; Ketumbar <br>
                &#10004; Rami <br>
                &#10004; Serat <br>
                &#10004; Teh <br>
                &#10004; Tebu <br>
                &#10004; Zaitun </p>
                """,
                unsafe_allow_html=True
            )
        st.markdown(
        "<h1 style='text-align: center;'>TANAMAN SEMUSIM</h1>"
        "<p style='text-align: center;'>Tanaman perkebunan tahunan adalah tanaman yang memiliki siklus hidup lebih dari <br>satu tahun dan memerlukan waktu lebih lama untuk tumbuh, berkembang, dan <br>menghasilkan hasil yang siap panen. Ini berbeda dengan tanaman perkebunan <br>semusim yang hanya hidup dalam satu musim pertumbuhan.</p>",
        unsafe_allow_html=True
        )
        col1, col2, col3 = st.columns([1, 4, 1])  # Membagi layout menjadi 3 bagian
        with col2:
            st.image('kelapa.jpg', use_column_width=True, output_format='JPEG')
        st.markdown(
        "<h1 style='text-align: center;'>TANAMAN TAHUNAN</h1>"
        "<p style='text-align: center;'>Tanaman perkebunan musiman adalah tanaman yang ditanam dan dipanen dalam satu <br>musim pertumbuhan tanaman. Mereka memiliki siklus hidup yang relatif singkat, <br>biasanya sekitar enam bulan hingga satu tahun, tergantung pada spesiesnya. Setelah <br>masa tanam yang relatif singkat, tanaman ini siap dipanen untuk dijual atau digunakan <br>untuk konsumsi.</p>",
        unsafe_allow_html=True
        )
        col1, col2, col3 = st.columns([1, 4, 1])  # Membagi layout menjadi 3 bagian
        with col2:
            st.image('teh.jpg', use_column_width=True, output_format='JPEG')
    elif menu == "Histori Data":
        st.markdown(
            "<h1 style='text-align: center;'>Data Histori</h1>",
            unsafe_allow_html=True
        )
        col1, col2, col3 = st.columns([1,  1, 1])  # Membagi layout menjadi 3 bagian
        with col2:
            st.write("Berikut adalah daftar negara beserta komoditas yang tersedia:")
            
            # Display dataframe across full width
            negara_komoditas_grouped = process_data()
            st.write(negara_komoditas_grouped, wide=True)

    elif menu == "Prediksi":
        col1, col2, col3 = st.columns([1,  1, 1])  # Membagi layout menjadi 3 bagian
        with col2:
            st.title("Prediksi Harga Komoditas")
        
            # Load data
            file_path = 'mean.csv'
            data = pd.read_csv(file_path, sep=';')
            
            # Group data by 'Komoditas' and 'Negara' and create regression models
            regression_models = data.groupby(['Komoditas', 'Negara'], group_keys=False).apply(lambda x: linear_regression(x)).to_dict()
            
            # Get unique list of komoditas from the data and capitalize each word
            komoditas_list = [komoditas.title() for komoditas in data['Komoditas'].unique()]
            
            # Allow user to select komoditas from the list
            komoditas_input = st.selectbox("Pilih Komoditas:", komoditas_list)
            volume_input = st.number_input("Masukkan Volume:", min_value=0.0, format="%.2f")
            tahun = st.selectbox("Pilih Tahun:", [2024, 2025, 2026])
            bulan = st.selectbox("Pilih Bulan:", list(range(1, 13)))

            if st.button("Predict"):
                table_data = []
                MIN_PREDICTED_PRICE = 0.01
                for key, (model, scaler) in regression_models.items():
                    additional_data = 0
                    predicted_prices = []
                    current_year, current_month = 2024, 1
                    for _ in range(36):
                        predicted_price = predict_price(model, additional_data, MIN_PREDICTED_PRICE, scaler)
                        predicted_prices.append(predicted_price)
                        additional_data = predicted_price
                        current_month += 1
                        if current_month > 12:
                            current_month = 1
                            current_year += 1
                    table_data.append([key[0], key[1], *predicted_prices])

                filtered_table_data = []
                for data_row in table_data:
                    komoditas, negara, *predicted_prices = data_row
                    if komoditas.title() == komoditas_input:  # Ensure matching with title-cased input
                        if tahun == 2024:
                            filtered_price = predicted_prices[bulan - 1]
                        elif tahun == 2025:
                            filtered_price = predicted_prices[12 + bulan - 1]
                        elif tahun == 2026:
                            filtered_price = predicted_prices[24 + bulan - 1]
                        else:
                            continue
                        harga_jual = round(filtered_price * volume_input, 1)
                        filtered_table_data.append([negara, f"{filtered_price:.1f}", f"{harga_jual:.1f}"])


                if filtered_table_data:
                    st.write("Hasil Prediksi:")
                    df = pd.DataFrame(filtered_table_data, columns=["Negara", "Predicted Price", "Harga Jual"])
                    st.table(df)
                else:
                    st.write("Data tidak ditemukan untuk komoditas yang dimasukkan.")
    elif menu == "Grafik Harga":
        col1, col2, col3 = st.columns([1, 1, 1])  # Membagi layout menjadi 3 bagian
        with col2:
            st.markdown(
                "<h1 style='text-align: center;'>Grafik Prediksi Harga</h1>"
                "<p style='text-align: center;'>Dalam grafik hasil prediksi, sumbu x ini biasanya mewakili nilai sebenarnya atau data yang diamati, sedangkan sumbu y mewakili nilai prediksi dari model atau metode yang digunakan. Garis ideal dalam grafik ini adalah garis diagonal dimana setiap titik pada grafik berada sepanjang garis tersebut, menunjukkan bahwa prediksi yang sempurna (nilai prediksi = nilai sebenarnya).</p>",
                unsafe_allow_html=True
            )
            st.markdown("<style> .stTextInput input {background-color: #FF0000 !important; color: white !important;} </style>", unsafe_allow_html=True)

            # Load data
            file_path = 'mean.csv'
            data = pd.read_csv(file_path, sep=';')

            # Get unique list of komoditas and negara from the data and capitalize each word
            komoditas_list = [komoditas.title() for komoditas in data['Komoditas'].unique()]
            negara_list = [negara.title() for negara in data['Negara'].unique()]

            # Allow user to select komoditas and negara from the list
            komoditas = st.selectbox("Pilih Komoditas:", komoditas_list)
            negara = st.selectbox("Pilih Negara:", negara_list)

            if st.button("Hitung Prediksi Harga"):
                if not generate_graph(komoditas.lower(), negara.lower(), file_path):
                    st.error("Data tidak ditemukan.")
    elif menu == "Grafik Linear":
        # Menyisipkan teks di tengah
        st.markdown(
            "<h1 style='text-align: center;'>Aplikasi Prediksi Regresi Linear</h1>"
            "<p style='text-align: center;'>Grafik regresi adalah representasi visual dari hubungan antara dua atau lebih variabel di dalam data. <br>Tujuan utama dari grafik regresi adalah untuk menunjukkan pola atau tren dalam data <br>dan memprediksi nilai variabel dependen berdasarkan variabel independen.</p>",
            unsafe_allow_html=True
        )

        # Membuat bagian kosong di sisi kiri dan kanan
        col1, col2, col3 = st.columns([1, 1, 1])  # Membagi layout menjadi 3 bagian
        with col2:
            st.markdown("<style> .stTextInput input {background-color: #FF0000 !important; color: white !important;} </style>", unsafe_allow_html=True)

            # Load data
            file_path = 'mean.csv'
            data = pd.read_csv(file_path, sep=';')

            # Get unique list of komoditas and negara from the data and capitalize each word
            komoditas_list = [komoditas.title() for komoditas in data['Komoditas'].unique()]
            negara_list = [negara.title() for negara in data['Negara'].unique()]

            # Allow user to select komoditas and negara from the list
            komoditas = st.selectbox("Pilih Komoditas:", komoditas_list)
            negara = st.selectbox("Pilih Negara:", negara_list)

            if st.button("Hitung Regresi Linear"):
                equation, graph_url = linear_regression_plot(komoditas.lower(), negara.lower())

                if equation:
                    st.write(f"Persamaan Regresi Linear: {equation}")
                    if graph_url:
                        st.image(io.BytesIO(base64.b64decode(graph_url)), use_column_width=True)
                else:
                    st.write(f"Tidak ada data yang cocok untuk komoditas {komoditas} di {negara}.")

    # CSS for the red footer
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: red;
            color: white;
            text-align: center;
            padding: 2px 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # HTML for the footer
    st.markdown(
        """
        <div class="footer">
            <p>&copy; 2024 Anggia Putri Wulan</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
