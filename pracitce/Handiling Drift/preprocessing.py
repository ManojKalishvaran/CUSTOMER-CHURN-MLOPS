from sklearn.base import BaseEstimator, TransformerMixin

class Preprocess(BaseEstimator, TransformerMixin): 
    def fit(self, x, y=None):
        return self

    def transform(self,x):
        x = x.copy()

        enc_gend = {"Male":0, "Female":1}
        x['Gender'] = x['Gender'].map(enc_gend)

        enc_sub = {'Premium':2, 'Standard':1, 'Basic':0}
        x['Subscription Type'] = x['Subscription Type'].map(enc_sub)

        enc_cont = {'Monthly':0, 'Quarterly':1, 'Annual':2}
        x['Contract Length'] = x['Contract Length'].map(enc_cont)

        return x