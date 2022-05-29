from Dataset.dataset import Eve
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
import pickle
from model import XSurvKMeansRF

ds = Eve('Dataset/Eve_data.csv')

(x_train, ye_train, y_train, e_train,
 x_val, ye_val, y_val, e_val,
 x_test, ye_test, y_test, e_test) = ds.get_train_val_test_from_splits(test_id=0, val_id=1)
 #train_df_orig, val_df_orig, test_df_orig) = ds.get_train_val_test_from_splits(test_id=0, val_id=1)

print('Testing on %d-----------------------------' % 1)
print(x_train.shape, x_val.shape)

# special for RSF
dt = np.dtype('bool,float')
y_train_surv = np.array([(bool(e), y) for e, y in zip(e_train, y_train)], dtype=dt)
y_val_surv = np.array([(bool(e), y) for e, y in zip(e_val, y_val)], dtype=dt)
print(y_train_surv.shape, y_val_surv.shape)

# train RSF
rsf = RandomSurvivalForest(n_estimators=50,
                           min_samples_split=5,
                           min_samples_leaf=5,
                           max_features="sqrt",
                           oob_score=True,
                           n_jobs=-1,
                           random_state=20)
# rsf.fit(x_train, y_train_surv)
# pickle.dump(rsf, open('rsf_model_eve.mdl', 'wb'))

rsf = pickle.load(open('rsf_model_eve.mdl', 'rb'))

cindex_train = rsf.score(x_train, y_train_surv)
cindex_oob = rsf.oob_score_
cindex_val = rsf.score(x_val, y_val_surv)
cindex_val_events = rsf.score(x_val[e_val==1], y_val_surv[e_val==1])

print('Train cindex {:.2f}'.format(cindex_train*100))
print('Test cindex {:.2f}'.format(cindex_val*100))
print('Test cindex Events Only {:.2f}'.format(cindex_val_events*100))

surv_train = rsf.predict_survival_function(x_train, return_array=True)
surv_val = rsf.predict_survival_function(x_val, return_array=True)
surv_test = rsf.predict_survival_function(x_test, return_array=True)



#Explanation
xte_data = (x_train, y_train, e_train,
            x_val, y_val, e_val,
            x_test, y_test, e_test)

survival_curves = (surv_train, surv_val, surv_test)

explainer = XSurvKMeansRF(max_k=30, z_explained_variance_ratio_threshold=0.99, curves_diff_significance_level=0.05)
explainer.fit(xte_data=xte_data, survival_curves=survival_curves, event_times=rsf.event_times_, pretrained_clustering_model="clustering_model_eve.mdl", k=9)
explainer.explain(x=x_train, features_names_list=ds.features_names)
explainer.explain(x=x_val, features_names_list=ds.features_names)
explainer.explain(x=x_test, features_names_list=ds.features_names)
