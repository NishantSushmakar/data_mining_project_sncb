from sklearn.naive_bayes import MultinomialNB, GaussianNB
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


models = {
    'mnb':MultinomialNB(),
    'gnb':GaussianNB(),
    'rf':RandomForestClassifier(n_estimators=1000),
    'lgbm':lgb.LGBMClassifier(n_estimator=10000, verbose=-1),
    'xgb': XGBClassifier(n_estimators=100, learning_rate=0.1),
    'lr':LogisticRegression(random_state=42, C=1.0, solver='lbfgs', max_iter=1000),
    'lgbm_over': lgb.LGBMClassifier(
                # Regularization parameters
                n_estimators=1000,  # High number of trees
                learning_rate=0.01,  # Lower learning rate to prevent overfitting
                
                # Regularization techniques
                regularization_factor=0.1,  # L2 regularization
                max_depth=7,  # Limit tree depth
                min_child_samples=20,  # Minimum number of samples in leaf
                
                # Prevent overfitting
                early_stopping_rounds=50,  # Stop if no improvement
                subsample=0.8,  # Take 80% of data for each tree
                colsample_bytree=0.8,  # Take 80% of features for each tree
                
                # Reduce model complexity
                num_leaves=31,  # Limit number of leaves
                
                # Prevent learning noise
                feature_fraction=0.7,  # Randomly select features
                bagging_fraction=0.7,  # Randomly select data
                bagging_freq=1,  # How often to perform bagging
                
                # Reduce verbosity
                verbose=-1
                )

 
}
