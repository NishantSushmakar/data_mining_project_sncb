from sklearn.naive_bayes import MultinomialNB, GaussianNB
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

models = {
    'mnb':MultinomialNB(),
    'gnb':GaussianNB(),
    'rf':RandomForestClassifier(n_estimators=1000),
    'lgbm':lgb.LGBMClassifier(n_estimator=50, verbose=-1)
    
}
