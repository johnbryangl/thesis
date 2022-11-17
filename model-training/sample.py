import pickle

model = pickle.load(open('rf_model.sav', 'rb'))
predictions = model.predict_proba([
    [794, 68.1, 235.34, 19.4, 100.23],
    [499, 24.73, 723.19, 15.00, 100.2]
])
print([1,2,3] in [1,2,3,4,5])