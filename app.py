from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

@app.route("/evaluate", methods=['GET', 'POST'])
def evaluate():
    if request.method == 'POST':
        csv_file = request.files['csv_file']
    return render_template('evaluate.html')

def evaluate_data(csv_file):
    try:
        # load csv in pandas dataframe
        lms_data = pd.read_csv(csv_file, sep=';')
        features = ['access', 'tests', 'exam', 'project', 'assignments']
        target = ['graduate']

        # validate csv
        for feature in features:
            if feature not in features:
                raise Exception(f'Column ({feature}) does not exist.')

        x = lms_data[features]
        y = lms_data[target]
    except:
        print('Error')

    return ''

if __name__ == '__main__':
    # initialize models here
    app.run(debug=True)
