import os
from flask import Flask, render_template, request, send_file
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload_pages():
    return render_template('upload.html')

@app.route('/proses', methods=['POST'])
def proses():
    min_support = float(request.form['min_support'])
    min_confidence = float(request.form['min_confidence'])

    file = request.files['file']
    if file.filename == '':
        return 'File tidak ditemukan!'

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    dataset = pd.read_csv(filepath)

    basket = (dataset.groupby(['Transaksi_ID', 'Item'])['Item']
              .count().unstack().reset_index().fillna(0).set_index('Transaksi_ID'))

    def encode_units(x):
        return 0 if x <= 0 else 1

    basket_sets = basket.applymap(encode_units)

    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)

    if frequent_itemsets.empty:
        return "Tidak ada item yang memenuhi minimum support."

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    if rules.empty:
        return "Tidak ada aturan yang memenuhi minimum confidence."

    rules_display = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

    rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules_display['consequents'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'hasil_rules.csv')
    rules_display.to_csv(output_path, index=False)

    deskripsi = []
    for _, row in rules_display.iterrows():
        desc = f"Jika membeli {row['antecedents']}, maka kemungkinan membeli {row['consequents']} adalah {row['confidence'] * 100:.2f}% (Lift: {row['lift']:.2f})."
        deskripsi.append(desc)

    return render_template('result.html', rules=rules_display.to_dict(orient='records'), deskripsi=deskripsi, file='hasil_rules.csv')

@app.route('/download/<filename>')
def download(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
