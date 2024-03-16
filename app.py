from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import os
from src.pipeline.predict_pipeline import CustomDataClass, Prediction
df = pd.read_csv('notebook/data/Clean_Laptop.csv')
app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'GET':
        company = sorted(df['Company'].unique())
        type_name = sorted(df['TypeName'].unique())
        ram = sorted(df['Ram'].unique())
        # weight = sorted(df['Weight'].unique())
        touch_screen = sorted(df['TouchScreen'].unique())
        ips = sorted(df['Ips'].unique())
        # ppi = sorted(df['ppi'].unique())
        cpu_brand = sorted(df['Cpu Brand'].unique())
        hdd = sorted(df['HDD'].unique())
        ssd = sorted(df['SSD'].unique())
        gpu_brand = sorted(df['Gpu Brand'].unique())
        os = sorted(df['OS'].unique())
        return render_template('index.html', companies = company, type_names = type_name, rams = ram, touch_screens = touch_screen, ips_s = ips,cpu_brands = cpu_brand,hdd_s = hdd, ssd_s = ssd, gpu_brands = gpu_brand, os_s = os)
    
    else:
        user_data = CustomDataClass(
            company = request.form.get('brand'),
            type_name = request.form.get('type'),
            ram = int(request.form.get('ram')),
            weight = float(request.form.get('weight')),
            touch_screen = int(request.form.get('touch')),
            ips = int(request.form.get('ips')),
            ppi = float(request.form.get('s_size')),
            cpu_brand = request.form.get('cpu'),
            hdd = int(request.form.get('hdd')),
            ssd = int(request.form.get('ssd')),
            gpu_brand = request.form.get('gpu'),
            os = request.form.get('os')
            )
        data_df = user_data.data_as_df()
        prediction_obj = Prediction()
        prediction_value = prediction_obj.prediction(data_df)

        return render_template('index.html', prediction = np.round(prediction_value[0],0))
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 8080, debug = True)

# port=int(os.environ.get('PORT', 80))