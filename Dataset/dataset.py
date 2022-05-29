'''
Fixed dataset classes (SurvEvo + adding Eve)
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
from itertools import permutations
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random


class Dataset:

    def __init__(self, dataset_file_path=None, number_of_splits=5,
                 drop_percentage=0, events_only=True, drop_feature=None,
                 random_seed=20, drop_corr_level=None):

        self.random_seed = random_seed
        self.dataset_file_path = dataset_file_path
        self.number_of_splits = number_of_splits
        self.drop_percentage = drop_percentage
        self.events_only = events_only
        self.drop_feature = drop_feature
        self.drop_corr_level = drop_corr_level
        #self.df = self._load_data()
        self._load_data()
        #self.n_splits = self._get_n_splits(seed=random_seed)
        self._get_n_splits(seed=random_seed)

        self.features_names = list(self.df.columns.drop(['T', 'E']))


        self.print_dataset_summery()

    @staticmethod
    def _get_corrolated_columns(ohdf, corr_level=0.999):
        # Dropping corrolated features
        cor = ohdf.corr()
        cor = cor[(cor >= corr_level) | (cor <= -corr_level)]
        cor = cor.dropna(thresh=2).dropna(how='all', axis=1).fillna(0)

        cols_to_drop = []
        for col in cor.columns:
            # print('Col:', col)
            # print('Dropped Cols', cols_to_drop)
            # print(not (col in cols_to_drop))
            if not (col in cols_to_drop):
                cor_cols = cor.loc[cor[col] != 0, [col]].index.to_list()
                print('Col:', col)
                print('Corrolated with:', cor_cols)
                cor_cols.remove(col)
                # print(cor_cols)
                cols_to_drop.extend(cor_cols)
        for col in ['T', 'E']:
            if col in cols_to_drop: cols_to_drop.remove(col)
        return cols_to_drop

    def get_dataset_name(self):
        pass

    def _preprocess_x(self, x_df):
        pass

    def _preprocess_y(self, y_df, normalizing_val=None):
        pass

    def _preprocess_e(self, e_df):
        pass

    def _fill_missing_values(self, x_train_df, x_val_df, x_test_df=None, x_tune_df=None):
        pass

    def _load_data(self):
        pass

    def get_x_dim(self):
        return self.df.shape[1]-2

    def _scale_x(self, x_train_df, x_val_df, x_test_df=None, x_tune_df=None):
        if (x_test_df is not None) & (x_tune_df is not None):
            return x_train_df.to_numpy(), x_val_df.to_numpy(), x_test_df.to_numpy(), x_tune_df.to_numpy()
        elif x_test_df is not None:
            return x_train_df.to_numpy(), x_val_df.to_numpy(), x_test_df.to_numpy()
        else:
            return x_train_df.to_numpy(), x_val_df.to_numpy()

    def print_dataset_summery(self):
        s = 'Dataset Description =======================\n'
        s += 'Dataset Name: {}\n'.format(self.get_dataset_name())
        s += 'Dataset Shape: {}\n'.format(self.df.shape)
        s += 'Events: %.2f %%\n' % (self.df['E'].sum()*100 / len(self.df))
        s += 'NaN Values: %.2f %%\n' % (self.df.isnull().sum().sum()*100 / self.df.size)
        s += f'Events % in splits: '
        for split in self.n_splits:
            s += '{:.2f}, '.format((split["E"].mean()*100))
        s += '\n'
        s += '===========================================\n'
        print(s)
        return s

    @staticmethod
    def max_transform(df, cols, powr):
        df_transformed = df.copy()
        for col in cols:
            df_transformed[col] = ((df_transformed[col]) / df_transformed[col].max()) ** powr
        return df_transformed

    @staticmethod
    def log_transform(df, cols):
        df_transformed = df.copy()
        for col in cols:
            df_transformed[col] = np.abs(np.log(df_transformed[col] + 1e-8))
        return df_transformed

    @staticmethod
    def power_transform(df, cols, powr):
        df_transformed = df.copy()
        for col in cols:
            df_transformed[col] = df_transformed[col] ** powr
        return df_transformed

    def _get_n_splits(self, seed=20):
        k = self.number_of_splits
        train_df = self.df
        df_splits = []
        for i in range(k, 1, -1):
            train_df, test_df = train_test_split(train_df, test_size=(1 / i), random_state=seed, shuffle=True,
                                                 stratify=train_df['E'])
            df_splits.append(test_df)
            if i == 2:
                df_splits.append(train_df)
        self.n_splits = df_splits
        #return df_splits

    def get_train_val_test_from_splits(self, val_id, test_id):
        df_splits_temp = self.n_splits.copy()
        val_df = df_splits_temp[val_id]
        test_df = df_splits_temp[test_id]
        train_df_splits = [df_splits_temp[i] for i in range(len(df_splits_temp)) if i not in [val_id, test_id]]
        train_df = pd.concat(train_df_splits)

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        x_train_df, y_train_df, e_train_df = self._split_columns(train_df)
        x_val_df, y_val_df, e_val_df = self._split_columns(val_df)
        x_test_df, y_test_df, e_test_df = self._split_columns(test_df)

        self._fill_missing_values(x_train_df, x_val_df, x_test_df)

        x_train, x_val, x_test = self._preprocess_x(x_train_df), \
                                 self._preprocess_x(x_val_df), \
                                 self._preprocess_x(x_test_df)

        x_train, x_val, x_test = self._scale_x(x_train, x_val, x_test)

        y_normalizing_val = y_train_df.max()

        y_train, y_val, y_test = self._preprocess_y(y_train_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_val_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_test_df, normalizing_val=y_normalizing_val)

        e_train, e_val, e_test = self._preprocess_e(e_train_df), \
                                 self._preprocess_e(e_val_df), \
                                 self._preprocess_e(e_test_df)

        ye_train, ye_val, ye_test = np.array(list(zip(y_train, e_train))), \
                                    np.array(list(zip(y_val, e_val))), \
                                    np.array(list(zip(y_test, e_test)))

        return (x_train, ye_train, y_train, e_train,
                x_val, ye_val, y_val, e_val,
                x_test, ye_test, y_test, e_test)

    def get_val_test_train_exclude_one_from_splits(self, val_id, test_id, excluded_id):
        df_splits_temp = self.n_splits.copy()
        val_df = df_splits_temp[val_id]
        test_df = df_splits_temp[test_id]
        train_df_splits = [df_splits_temp[i] for i in range(len(df_splits_temp)) if i not in [val_id, test_id, excluded_id]]
        train_df = pd.concat(train_df_splits)

        x_train_df, y_train_df, e_train_df = self._split_columns(train_df)
        x_val_df, y_val_df, e_val_df = self._split_columns(val_df)
        x_test_df, y_test_df, e_test_df = self._split_columns(test_df)

        self._fill_missing_values(x_train_df, x_val_df, x_test_df)

        x_train, x_val, x_test = self._preprocess_x(x_train_df), \
                                 self._preprocess_x(x_val_df), \
                                 self._preprocess_x(x_test_df)

        x_train, x_val, x_test = self._scale_x(x_train, x_val, x_test)

        y_normalizing_val = y_train_df.max()

        y_train, y_val, y_test = self._preprocess_y(y_train_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_val_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_test_df, normalizing_val=y_normalizing_val)

        e_train, e_val, e_test = self._preprocess_e(e_train_df), \
                                 self._preprocess_e(e_val_df), \
                                 self._preprocess_e(e_test_df)

        ye_train, ye_val, ye_test = np.array(list(zip(y_train, e_train))), \
                                    np.array(list(zip(y_val, e_val))), \
                                    np.array(list(zip(y_test, e_test)))

        return (x_train, ye_train, y_train, e_train,
                x_val, ye_val, y_val, e_val,
                x_test, ye_test, y_test, e_test)

    def get_val_test_sampled_train_from_splits(self, val_id, test_id, frac=0.8, replace=True, seed=20):
        df_splits_temp = self.n_splits.copy()
        val_df = df_splits_temp[val_id]
        test_df = df_splits_temp[test_id]
        train_df_splits = [df_splits_temp[i] for i in range(len(df_splits_temp)) if i not in [val_id, test_id]]
        train_df = pd.concat(train_df_splits)

        sampled_train_df = train_df.sample(frac=frac, replace=replace, random_state=seed)

        x_train_df, y_train_df, e_train_df = self._split_columns(sampled_train_df)
        x_val_df, y_val_df, e_val_df = self._split_columns(val_df)
        x_test_df, y_test_df, e_test_df = self._split_columns(test_df)

        self._fill_missing_values(x_train_df, x_val_df, x_test_df)

        x_train, x_val, x_test = self._preprocess_x(x_train_df), \
                                 self._preprocess_x(x_val_df), \
                                 self._preprocess_x(x_test_df)

        x_train, x_val, x_test = self._scale_x(x_train, x_val, x_test)

        y_normalizing_val = y_train_df.max()

        y_train, y_val, y_test = self._preprocess_y(y_train_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_val_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_test_df, normalizing_val=y_normalizing_val)

        e_train, e_val, e_test = self._preprocess_e(e_train_df), \
                                 self._preprocess_e(e_val_df), \
                                 self._preprocess_e(e_test_df)

        ye_train, ye_val, ye_test = np.array(list(zip(y_train, e_train))), \
                                    np.array(list(zip(y_val, e_val))), \
                                    np.array(list(zip(y_test, e_test)))

        return (x_train, ye_train, y_train, e_train,
                x_val, ye_val, y_val, e_val,
                x_test, ye_test, y_test, e_test)

    def get_train_val_test_tune_from_splits(self, val_id, test_id, tune_id):
        df_splits_temp = self.n_splits.copy()
        val_df = df_splits_temp[val_id]
        tune_df = df_splits_temp[tune_id]
        test_df = df_splits_temp[test_id]
        train_df_splits = [df_splits_temp[i] for i in range(len(df_splits_temp)) if i not in [val_id, test_id, tune_id]]
        train_df = pd.concat(train_df_splits)

        x_train_df, y_train_df, e_train_df = self._split_columns(train_df)
        x_val_df, y_val_df, e_val_df = self._split_columns(val_df)
        x_test_df, y_test_df, e_test_df = self._split_columns(test_df)
        x_tune_df, y_tune_df, e_tune_df = self._split_columns(tune_df)

        self._fill_missing_values(x_train_df, x_val_df, x_test_df, x_tune_df)

        x_train, x_val, x_test, x_tune = self._preprocess_x(x_train_df), \
                                         self._preprocess_x(x_val_df), \
                                         self._preprocess_x(x_test_df), \
                                         self._preprocess_x(x_tune_df)

        x_train, x_val, x_test = self._scale_x(x_train, x_val, x_test)

        y_normalizing_val = y_train_df.max()

        y_train, y_val, y_test, y_tune = self._preprocess_y(y_train_df, normalizing_val=y_normalizing_val), \
                                         self._preprocess_y(y_val_df, normalizing_val=y_normalizing_val), \
                                         self._preprocess_y(y_test_df, normalizing_val=y_normalizing_val), \
                                         self._preprocess_y(y_tune_df, normalizing_val=y_normalizing_val)

        e_train, e_val, e_test, e_tune = self._preprocess_e(e_train_df), \
                                         self._preprocess_e(e_val_df), \
                                         self._preprocess_e(e_test_df), \
                                         self._preprocess_e(e_tune_df)

        ye_train, ye_val, ye_test, ye_tune = np.array(list(zip(y_train, e_train))), \
                                             np.array(list(zip(y_val, e_val))), \
                                             np.array(list(zip(y_test, e_test))), \
                                             np.array(list(zip(y_tune, e_tune)))

        return (x_train, ye_train, y_train, e_train,
                x_val, ye_val, y_val, e_val,
                x_test, ye_test, y_test, e_test,
                x_tune, ye_tune, y_tune, e_tune)


    @staticmethod
    def get_shuffled_pairs(x, y, e, seed=None):
        x_sh, y_sh, e_sh = shuffle(x, y, e, random_state=seed)
        y_diff = y_sh - y
        fltr = (e == 1) & (y_diff > 0)  # choose the first item in the pair to be an event
        return x[fltr], y[fltr], x_sh[fltr], y_sh[fltr], y_diff[fltr]

    def get_train_val_from_splits(self, val_id):
        df_splits_temp = self.n_splits.copy()
        val_df = df_splits_temp[val_id]
        train_df_splits = [df_splits_temp[i] for i in range(len(df_splits_temp)) if i not in [val_id]]
        train_df = pd.concat(train_df_splits)

        x_train_df, y_train_df, e_train_df = self._split_columns(train_df)
        x_val_df, y_val_df, e_val_df = self._split_columns(val_df)

        self._fill_missing_values(x_train_df, x_val_df)

        x_train, x_val = self._preprocess_x(x_train_df), self._preprocess_x(x_val_df)

        x_train, x_val = self._scale_x(x_train, x_val)

        y_train, y_val = self._preprocess_y(y_train_df), self._preprocess_y(y_val_df)

        e_train, e_val = self._preprocess_e(e_train_df), self._preprocess_e(e_val_df)

        ye_train, ye_val = np.array(list(zip(y_train, e_train))), np.array(list(zip(y_val, e_val)))

        # TODO: Add the y_surv to the output (t,e), suitable for RSF

        return (x_train, ye_train, y_train, e_train,
                x_val, ye_val, y_val, e_val)

    @staticmethod
    def _split_columns(df):
        y_df = df['T']
        e_df = df['E']
        x_df = df.drop(['T', 'E'], axis=1)
        return x_df, y_df, e_df

    def test_dataset(self):
        combs = list(permutations(range(self.number_of_splits), 2))
        for i, j in combs:
            (x_train, ye_train, y_train, e_train,
             x_val, ye_val, y_val, e_val,
             x_test, ye_test, y_test, e_test) = self.get_train_val_test_from_splits(i, j)
            assert np.isnan(x_train).sum() == 0
            assert np.isnan(x_val).sum() == 0
            assert np.isnan(x_test).sum() == 0


class Metabric(Dataset):
    def _load_data(self):
        base_file_name = '.'.join(self.dataset_file_path.split('.')[:-1])
        xdf = pd.read_csv(base_file_name+'.csv')
        ydf = pd.read_csv(base_file_name+'_label.csv')
        ydf.rename(columns={'event_time': 'T', 'label': 'E'}, inplace=True)
        df = xdf.join(ydf)
        self.df = df
        #return df

    def get_dataset_name(self):
        return 'METABRIC'

    def _preprocess_x(self, x_df):
        return super().max_transform(x_df,
                                     ['age_at_diagnosis', 'size', 'lymph_nodes_positive', 'stage',
                                      'lymph_nodes_removed', 'NPI'], 1)

    def _preprocess_y(self, y_df, normalizing_val=None):
        if normalizing_val is None:
            normalizing_val = y_df.max()
        return ((y_df / normalizing_val).to_numpy() ** 0.5).astype('float32')

    def _preprocess_e(self, e_df):
        return e_df.to_numpy().astype('float32')


class Flchain(Dataset):
    def _load_data(self):
        df = pd.read_csv(self.dataset_file_path, index_col='idx')
        df['sex'] = df['sex'].map(lambda x: 0 if x == 'M' else 1)
        df.drop('chapter', axis=1, inplace=True)
        df['sample.yr'] = df['sample.yr'].astype('category')
        df['flc.grp'] = df['flc.grp'].astype('category')
        df.rename(columns={'futime': 'T', 'death': 'E'}, inplace=True)
        ohdf = pd.get_dummies(df)
        self.df = ohdf
        #return ohdf

    def get_dataset_name(self):
        return 'flchain'

    def _fill_missing_values(self, x_train_df, x_val_df, x_test_df=None, x_tune_df=None):
        m = x_train_df['creatinine'].median()
        x_train_df['creatinine'].fillna(m, inplace=True)
        x_val_df['creatinine'].fillna(m, inplace=True)
        if x_test_df is not None:
            x_test_df['creatinine'].fillna(m, inplace=True)
        if x_tune_df is not None:
            x_tune_df['creatinine'].fillna(m, inplace=True)

    def _preprocess_x(self, x_df):
        return x_df

    def _preprocess_y(self, y_df, normalizing_val=None):
        if normalizing_val is None:
            normalizing_val = y_df.max()
        return ((y_df / normalizing_val).to_numpy() ** 0.5).astype('float32')

    def _preprocess_e(self, e_df):
        return e_df.to_numpy().astype('float32')

    def _scale_x(self, x_train_df, x_val_df, x_test_df=None, x_tune_df=None):
        scaler = StandardScaler().fit(x_train_df)
        x_train = scaler.transform(x_train_df)
        x_val = scaler.transform(x_val_df)
        if (x_tune_df is not None) & (x_test_df is not None):
            x_test = scaler.transform(x_test_df)
            x_tune = scaler.transform(x_tune_df)
            return x_train, x_val, x_test, x_tune
        elif x_test_df is not None:
            x_test = scaler.transform(x_test_df)
            return x_train, x_val, x_test
        else:
            return x_train, x_val

class FlchainSub(Flchain):
    def _load_data(self):
        df = pd.read_csv(self.dataset_file_path, index_col='idx')
        df['sex'] = df['sex'].map(lambda x: 0 if x == 'M' else 1)
        df['SigmaFLC'] = df['kappa'] + df['lambda']
        df['rFLC'] = df['kappa'] / df['lambda']
        df.drop(['chapter', 'sample.yr', 'flc.grp', 'kappa', 'lambda', 'mgus'], axis=1, inplace=True)

        #df['sample.yr'] = df['sample.yr'].astype('category')
        #df['flc.grp'] = df['flc.grp'].astype('category')
        df.rename(columns={'futime': 'T', 'death': 'E'}, inplace=True)
        ohdf = pd.get_dummies(df)
        self.df = ohdf
        #return ohdf

    def get_dataset_name(self):
        return 'flchain_sub'

class FlchainSub1(Flchain):
    def _load_data(self):
        df = pd.read_csv(self.dataset_file_path, index_col='idx')
        df['sex'] = df['sex'].map(lambda x: 0 if x == 'M' else 1)
        df['SigmaFLC'] = df['kappa'] + df['lambda']
        df.drop(['chapter', 'sample.yr', 'flc.grp', 'kappa', 'lambda', 'mgus'], axis=1, inplace=True)

        #df['sample.yr'] = df['sample.yr'].astype('category')
        #df['flc.grp'] = df['flc.grp'].astype('category')
        df.rename(columns={'futime': 'T', 'death': 'E'}, inplace=True)
        ohdf = pd.get_dummies(df)
        self.df = ohdf
        #return ohdf

    def get_dataset_name(self):
        return 'flchain_sub1'

class Nwtco(Dataset):
    def _load_data(self):
        df = pd.read_csv(self.dataset_file_path, )
        df.drop(columns=['idx'], inplace=True)
        df = (df.assign(instit_2=df['instit'] - 1,
                        histol_2=df['histol'] - 1,
                        study_4=df['study'] - 3,
                        stage=df['stage'].astype('category'))
              .drop(['seqno', 'instit', 'histol', 'study'], axis=1))
        for col in df.columns.drop('stage'):
            df[col] = df[col].astype('float32')
        df.rename(columns={'edrel': 'T', 'rel': 'E'}, inplace=True)
        ohdf = pd.get_dummies(df)
        self.df = ohdf
        #return ohdf

    def get_dataset_name(self):
        return 'Nwtco'

    def _preprocess_x(self, x_df):
        return x_df

    def _preprocess_y(self, y_df, normalizing_val=None):
        if normalizing_val is None:
            normalizing_val = y_df.max()
        return ((y_df / normalizing_val).to_numpy() ** 0.5).astype('float32')

    def _preprocess_e(self, e_df):
        return e_df.to_numpy().astype('float32')

    def _scale_x(self, x_train_df, x_val_df, x_test_df=None, x_tune_df=None):
        scaler = StandardScaler().fit(x_train_df)
        x_train = scaler.transform(x_train_df)
        x_val = scaler.transform(x_val_df)
        if (x_tune_df is not None) & (x_test_df is not None):
            x_test = scaler.transform(x_test_df)
            x_tune = scaler.transform(x_tune_df)
            return x_train, x_val, x_test, x_tune
        elif x_test_df is not None:
            x_test = scaler.transform(x_test_df)
            return x_train, x_val, x_test
        else:
            return x_train, x_val


class Support(Dataset):
    def _load_data(self):
        df = pd.read_csv(self.dataset_file_path, index_col=0)
        one_hot_encoder_list = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca', 'sfdm2']
        if self.drop_feature is None:
            to_drop = ['hospdead', 'prg2m', 'prg6m', 'dnr', 'dnrday', 'aps', 'sps', 'surv2m', 'surv6m', 'totmcst']
        else:
            to_drop = ['hospdead', 'prg2m', 'prg6m', 'dnr', 'dnrday', 'aps', 'sps', 'surv2m', 'surv6m', 'totmcst'] + [self.drop_feature]
            if self.drop_feature in one_hot_encoder_list:
                one_hot_encoder_list.remove(self.drop_feature)

        df.drop(columns=to_drop, inplace=True)
        df.rename(columns={'d.time': 'T', 'death': 'E'}, inplace=True)
        ohdf = pd.get_dummies(df, prefix=one_hot_encoder_list, columns=one_hot_encoder_list)
        self.encoded_indices = self.one_hot_indices(ohdf, one_hot_encoder_list)

        if self.drop_percentage > 0:
            if self.events_only:
                number_of_events = ohdf['E'].sum()
                drop_n = int(self.drop_percentage * number_of_events)
                idxs = list(ohdf[(ohdf['E'] == 1)].sample(drop_n, random_state=20).index)
                ohdf = ohdf[~ohdf.index.isin(idxs)]
            else:
                number_of_samples = ohdf.shape[0]
                drop_n = int(self.drop_percentage * number_of_samples)
                idxs = list(ohdf.sample(drop_n, random_state=20).index)
                ohdf = ohdf[~ohdf.index.isin(idxs)]
        self.df = ohdf
        #return ohdf

    def get_dataset_name(self):
        return 'support'

    def _preprocess_x(self, x_df):
        features = ['totcst', 'charges', 'pafi', 'sod']
        if ~(self.drop_feature is None):
            if self.drop_feature in features:
                features.remove(self.drop_feature)
        return super().log_transform(x_df, features)

    def _preprocess_y(self, y_df, normalizing_val=None):
        if normalizing_val is None:
            normalizing_val = y_df.max()
        return ((y_df / normalizing_val).to_numpy() ** 0.1).astype('float32')

    def _preprocess_e(self, e_df):
        return e_df.to_numpy().astype('float32')

    def _scale_x(self, x_train_df, x_val_df, x_test_df=None, x_tune_df=None):
        scaler = StandardScaler().fit(x_train_df)
        x_train = scaler.transform(x_train_df)
        x_val = scaler.transform(x_val_df)
        if (x_tune_df is not None) & (x_test_df is not None):
            x_test = scaler.transform(x_test_df)
            x_tune = scaler.transform(x_tune_df)
            return x_train, x_val, x_test, x_tune
        elif x_test_df is not None:
            x_test = scaler.transform(x_test_df)
            return x_train, x_val, x_test
        else:
            return x_train, x_val

    def _fill_missing_values(self, x_train_df, x_val_df, x_test_df=None, x_tune_df=None):
        imputation_values = self.get_train_median_mode(x=x_train_df.to_numpy(), categorial=self.encoded_indices)
        imputation_vals_dict = dict(zip(self.df.columns, imputation_values))
        x_train_df.fillna(imputation_vals_dict, inplace=True)
        x_val_df.fillna(imputation_vals_dict, inplace=True)
        if x_test_df is not None:
            x_test_df.fillna(imputation_vals_dict, inplace=True)
        if x_tune_df is not None:
            x_tune_df.fillna(imputation_vals_dict, inplace=True)

    @staticmethod
    def one_hot_indices(dataset, one_hot_encoder_list):
        """
        The function is copied from: https://github.com/paidamoyo/adversarial_time_to_event
        """
        indices_by_category = []
        for colunm in one_hot_encoder_list:
            values = dataset.filter(regex="{}_.*".format(colunm)).columns.values
            indices_one_hot = []
            for value in values:
                indice = dataset.columns.get_loc(value)
                indices_one_hot.append(indice)
            indices_by_category.append(indices_one_hot)
        return indices_by_category

    @staticmethod
    def get_train_median_mode(x, categorial):
        """
        The function is copied from: https://github.com/paidamoyo/adversarial_time_to_event
        """
        def flatten_nested(list_of_lists):
            flattened = [val for sublist in list_of_lists for val in sublist]
            return flattened

        categorical_flat = flatten_nested(categorial)
        imputation_values = []
        median = np.nanmedian(x, axis=0)
        mode = []
        for idx in np.arange(x.shape[1]):
            a = x[:, idx]
            (_, idx, counts) = np.unique(a, return_index=True, return_counts=True)
            index = idx[np.argmax(counts)]
            mode_idx = a[index]
            mode.append(mode_idx)
        for i in np.arange(x.shape[1]):
            if i in categorical_flat:
                imputation_values.append(mode[i])
            else:
                imputation_values.append(median[i])
        return imputation_values


class Eve(Dataset):
    def _load_data(self):
        df = pd.read_csv(self.dataset_file_path)
        df_orig = df.copy()
        self.split_on = df['CHASSIS_NO']
        to_drop = ['CHASSIS_NO']
        for col in to_drop:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
        one_hot_encoder_list = ['CHASSIS_SERIES', 'VEHICLE_TYPE', 'EMISSION_LEVEL', 'CONTRACT_TYPE',
                                'TRANSPORT_CYCLE', 'CITY', 'COUNTRY', 'CONTINENT']
        ohdf = pd.get_dummies(df, prefix=one_hot_encoder_list, columns=one_hot_encoder_list)

        if self.drop_corr_level is not None:
            cols_to_drop = self._get_corrolated_columns(ohdf, self.drop_corr_level)
            print(cols_to_drop)
            ohdf.drop(columns=cols_to_drop, inplace=True)
        if 'EMISSION_LEVEL_1' in ohdf.columns:
            ohdf.drop(columns=['EMISSION_LEVEL_1'], inplace=True)
        self.df = ohdf
        self.df_orig = df_orig

    def _get_n_splits(self, seed=20):
        chassis = self.split_on.unique()
        random.seed(seed)
        random.shuffle(chassis)
        step = int(len(chassis) * (1.0 / self.number_of_splits))
        ch_splits = []
        for i in range(self.number_of_splits):
            i1 = i * step
            if i < self.number_of_splits - 1:
                chi = chassis[i1:i1 + step].copy()
            else:
                chi = chassis[i1:].copy()
            ch_splits.append(chi)
        df_orig_splits = []
        df_splits = []
        for chi in ch_splits:
            dfi = self.df[self.split_on.isin(chi)]
            df_orig_i = self.df_orig[self.split_on.isin(chi)]
            df_splits.append(dfi)
            df_orig_splits.append(df_orig_i)

        self.n_splits = df_splits
        self.df_orig_splits = df_orig_splits
        #return df_splits, df_orig_splits

    def get_dataset_name(self):
        return 'Eve'

    def _preprocess_x(self, x_df):
        return x_df

    def _preprocess_y(self, y_df, normalizing_val=None):
        if normalizing_val is None:
            normalizing_val = y_df.max()
        return ((y_df / normalizing_val).to_numpy() ** 0.5).astype('float32')

    def _preprocess_e(self, e_df):
        return e_df.to_numpy().astype('float32')

    def _scale_x(self, x_train_df, x_val_df, x_test_df=None, x_tune_df=None):
        scaler = StandardScaler().fit(x_train_df)
        x_train = scaler.transform(x_train_df)
        x_val = scaler.transform(x_val_df)

        if (x_tune_df is not None) & (x_test_df is not None):
            x_test = scaler.transform(x_test_df)
            x_tune = scaler.transform(x_tune_df)
            return x_train, x_val, x_test, x_tune
        elif x_test_df is not None:
            x_test = scaler.transform(x_test_df)
            return x_train, x_val, x_test
        else:
            return x_train, x_val


class Bars3DCross(Dataset):
    def _load_data(self):

        n_features = 3
        t_base = {0: 0.1, 1: 0.5, 2: 1}
        x_stds = [0.1, 0.1, 0.1]
        x_dependencies = {0: [0, 1], 1: [1, 2], 2: [2, 0]}
        x_weights = {0: [0.7, 0.3], 1: [0.5, 0.5], 2: [0.3, 0.7]}
        x_transformations = {0: [SimData.I, SimData.I], 1: [SimData.I, SimData.I], 2: [SimData.I, SimData.I]}
        t_noises = [0.2, 0.05, 0.1]
        n_samples = np.array([3000, 3000, 3000])
        centers = np.array([[0, 0, 1],
                            [0, 1, 0],
                            [1, 1, 1]]
                           )
        sdata = SimDataUniform(n_samples=n_samples, n_centers=3, centers=centers, n_features=n_features, t_base=t_base, x_stds=x_stds,
                        x_transformations=x_transformations, x_dependencies=x_dependencies, x_weights=x_weights,
                        noise=t_noises, p_censoring=0.7, censoring_effect=1, random_state=self.random_seed)

        self.sdata = sdata

        features = [f'X{i}' for i in range(n_features)]
        ohdf = pd.DataFrame(sdata.x, columns=features)
        ohdf['T'] = sdata.t
        ohdf['E'] = sdata.e
        self.df = ohdf
        self.df_orig = ohdf

        sdata.plot_data()

        fig = plt.figure(figsize=(20, 7))
        ax = fig.add_subplot(121, projection='3d')

        ax.scatter(sdata.x[:, 0], sdata.x[:, 1], sdata.x[:, 2], c=sdata.getc(sdata.c), alpha=0.2)
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_zlabel('x2')

        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(sdata.x[:, 0], sdata.x[:, 1], sdata.x[:, 2], c=sdata.t, alpha=0.2)
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_zlabel('x2')
        #return ohdf

    def get_dataset_name(self):
        return 'Bars3D'

    def _preprocess_x(self, x_df):
        return x_df

    def _preprocess_y(self, y_df, normalizing_val=None):
        if normalizing_val is None:
            normalizing_val = y_df.max()
        return ((y_df / normalizing_val).to_numpy() ** 0.5).astype('float32')

    def _preprocess_e(self, e_df):
        return e_df.to_numpy().astype('float32')

    def _scale_x(self, x_train_df, x_val_df, x_test_df=None, x_tune_df=None):
        scaler = StandardScaler().fit(x_train_df)
        x_train = scaler.transform(x_train_df)
        x_val = scaler.transform(x_val_df)
        if (x_tune_df is not None) & (x_test_df is not None):
            x_test = scaler.transform(x_test_df)
            x_tune = scaler.transform(x_tune_df)
            return x_train, x_val, x_test, x_tune
        elif x_test_df is not None:
            x_test = scaler.transform(x_test_df)
            return x_train, x_val, x_test
        else:
            return x_train, x_val


class Points3D(Dataset):
    def _load_data(self):

        n_features = 3
        t_base = {0: 0.1, 1: 1, 2: 2}
        x_stds = [0.0, 0.0, 0.0]
        x_dependencies = {0: [0, 1], 1: [1, 2], 2: [2, 0]}
        x_weights = {0: [0.7, 0.3], 1: [0.5, 0.5], 2: [0.3, 0.7]}
        x_transformations = {0: [SimData.I, SimData.I], 1: [SimData.I, SimData.I], 2: [SimData.I, SimData.I]}
        t_noises = [0.1, 0.2, 0.3]
        n_samples = np.array([3000, 3000, 3000])
        centers = np.array([[0, 0, 1],
                            [0, 1, 0],
                            [1, 1, 1]]
                           )
        sdata = SimDataDiscrete(n_samples=n_samples, n_centers=3, centers=centers, n_features=n_features, t_base=t_base, x_stds=x_stds,
                        x_transformations=x_transformations, x_dependencies=x_dependencies, x_weights=x_weights,
                        noise=t_noises, p_censoring=0.7, censoring_effect=1, random_state=self.random_seed)

        self.sdata = sdata

        features = [f'X{i}' for i in range(n_features)]
        ohdf = pd.DataFrame(sdata.x, columns=features)
        ohdf['T'] = sdata.t
        ohdf['E'] = sdata.e
        self.df = ohdf
        self.df_orig = ohdf

        sdata.plot_data()

        fig = plt.figure(figsize=(20, 7))
        ax = fig.add_subplot(121, projection='3d')

        ax.scatter(sdata.x[:, 0], sdata.x[:, 1], sdata.x[:, 2], c=sdata.getc(sdata.c), alpha=0.2)
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_zlabel('x2')

        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(sdata.x[:, 0], sdata.x[:, 1], sdata.x[:, 2], c=sdata.t, alpha=0.2)
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_zlabel('x2')
        #return ohdf

    def get_dataset_name(self):
        return 'Points3D'

    def _preprocess_x(self, x_df):
        return x_df

    def _preprocess_y(self, y_df, normalizing_val=None):
        if normalizing_val is None:
            normalizing_val = y_df.max()
        #return ((y_df / normalizing_val).to_numpy() ** 0.5).astype('float32')
        return ((y_df / normalizing_val).to_numpy()).astype('float32')

    def _preprocess_e(self, e_df):
        return e_df.to_numpy().astype('float32')

    def _scale_x(self, x_train_df, x_val_df, x_test_df=None, x_tune_df=None):
        scaler = StandardScaler().fit(x_train_df)
        x_train = scaler.transform(x_train_df)
        x_val = scaler.transform(x_val_df)
        if (x_tune_df is not None) & (x_test_df is not None):
            x_test = scaler.transform(x_test_df)
            x_tune = scaler.transform(x_tune_df)
            return x_train, x_val, x_test, x_tune
        elif x_test_df is not None:
            x_test = scaler.transform(x_test_df)
            return x_train, x_val, x_test
        else:
            return x_train, x_val


class SimData:
    def __init__(self, n_samples, n_centers, centers, n_features, t_base, x_stds,
                 x_transformations, x_dependencies, x_weights,
                 p_censoring, censoring_effect, noise=0.1, random_state=0):
        self.n_samples = n_samples
        self.n_centers = n_centers
        self.centers = centers
        self.n_features = n_features
        self.t_base = t_base
        self.x_stds = x_stds
        self.x_transformations = x_transformations
        self.x_dependencies = x_dependencies
        self.x_weights = x_weights

        self.p_censoring = p_censoring
        self.censoring_effect = censoring_effect
        self.noise = noise
        self.random_state = random_state
        self.x, self.c, self.t, self.e, self.t_uncensored = self.get_data()

    def get_data(self):
        pass

    def get_te(self, x, c):
        np.random.seed(self.random_state)
        if isinstance(self.n_samples, int):
            n_samples = self.n_samples
        else:
            n_samples = sum(self.n_samples)

        t_uncensored = np.array([self.get_t_x(xi, ci) for xi, ci in zip(x, c)])
        t_uncensored += abs(t_uncensored.min())

        e = (np.random.random(n_samples) >= self.p_censoring).astype(int)

        censoring_fracs = self.censoring_effect * np.random.uniform(0 , 1, n_samples)
        t = np.array([ti - (ti * pi) if ei == 0 else ti for ti, pi, ei in zip(t_uncensored, censoring_fracs, e)])

        return t, e, t_uncensored

    def get_t_x(self, x, c):
        e = 1
        tc = self.t_base[c]

        fc = self.x_transformations[c]
        # print(self.x_dependencies[c])
        # print(x.shape)
        xc = x[self.x_dependencies[c]]
        fxc = np.array([fi(xi) for fi, xi in zip(fc, xc)])

        wc = self.x_weights[c]

        # tx = tc + np.dot(wc, fxc) + np.random.normal(0,self.noise)
        if isinstance(self.noise, int):
            noise = self.noise
        else:
            noise = self.noise[c]

        tx = tc + np.random.normal(0, noise)
        return tx

    @staticmethod
    def getc(c):
        return [f'C{i}' for i in c]

    @staticmethod
    def I(x):
        return x

    def plot_data(self, alpha=0.2):
        lines = self.n_centers + 1
        fig, ax = plt.subplots(lines, 4, figsize=(20, lines * 5))
        ax[0, 0].scatter(self.x[:, 0], self.x[:, 1], c=self.getc(self.c), alpha=alpha)
        ax[0, 0].set_title('Clusters Ground Truth')
        ax[0, 0].set_xlabel(f'x0')
        ax[0, 0].set_ylabel(f'x1')

        ax[0, 1].hist(self.t[self.e == 0], bins=50, alpha=0.7, label='Censored');
        ax[0, 1].hist(self.t[self.e == 1], bins=50, alpha=0.7, label='Events');
        ax[0, 1].set_xlabel('time')
        ax[0, 1].set_title('Event Times')
        ax[0, 1].legend()

        ax[0, 2].hist(self.t_uncensored, bins=50, alpha=0.7);
        ax[0, 2].set_xlabel('time')
        ax[0, 2].set_title('Actual Event Times')

        for i in range(self.n_centers):
            self._plot_center(i, ax[i + 1], alpha)

    def _plot_center(self, c, ax, alpha):
        idxs = self.x_dependencies[c]
        x = self.x[:, idxs]

        ax[0].scatter(x[:, 0], x[:, 1], c=self.getc(self.c), alpha=alpha)
        ax[0].set_title('Clusters')
        ax[0].set_xlabel(f'x{idxs[0]}')
        ax[0].set_ylabel(f'x{idxs[1]}')

        ax[1].scatter(x[self.e == 1, 0], x[self.e == 1, 1], c=self.t[self.e == 1], alpha=alpha)
        ax[1].set_title('Events')
        ax[1].set_xlabel(f'x{idxs[0]}')
        ax[1].set_ylabel(f'x{idxs[1]}')

        ax[2].scatter(x[self.e == 0, 0], x[self.e == 0, 1], c=self.t[self.e == 0], alpha=alpha)
        ax[2].set_title('Censored')
        ax[2].set_xlabel(f'x{idxs[0]}')
        ax[2].set_ylabel(f'x{idxs[1]}')

        ax[3].scatter(x[:, 0], x[:, 1], c=self.t_uncensored, alpha=alpha)
        ax[3].set_title('Actual Event Times')
        ax[3].set_xlabel(f'x{idxs[0]}')
        ax[3].set_ylabel(f'x{idxs[1]}')

    def save(self, file_name='sim_data.npy'):
        data = {'x': self.x, 'y': self.c, 't': self.t, 'e': self.e}
        np.save(file_name, data)


class SimDataUniform(SimData):
    def get_data(self):
        from sklearn.datasets import make_blobs
        np.random.seed(self.random_state)
        if isinstance(self.n_samples, int):
            centers = self.n_centers
        else:
            if self.centers is None:
                # np.random.seed(self.random_state)
                centers = (np.random.normal(size=(self.n_centers, self.n_features)))
            else:
                centers = self.centers

        x, c = make_blobs(n_samples=self.n_samples, centers=centers, cluster_std=self.x_stds,
                          n_features=self.n_features, random_state=self.random_state)

        for k in range(self.n_centers):
            idxs = self.x_dependencies[k]
            for i in range(self.n_features):
                if i not in idxs:
                    n = len(x[c == k, i])
                    xmin = x[:, i].min()
                    xmax = x[:, i].max()
                    x[c==k,i] = np.random.uniform(xmin, xmax, n)

        t, e, t_uncensored = self.get_te(x, c)
        return x, c, t, e, t_uncensored

    def get_t_x(self, x, c):
        e = 1
        tc = self.t_base[c]

        fc = self.x_transformations[c]
        xc = x[self.x_dependencies[c]]
        fxc = np.array([fi(xi) for fi, xi in zip(fc, xc)])

        wc = self.x_weights[c]

        if isinstance(self.noise, int):
            noise = self.noise
        else:
            noise = self.noise[c]

        tx = tc + 0.1*np.exp(np.dot(wc, fxc)) + np.random.normal(0, noise)
        return tx


class SimDataDiscrete(SimData):
    def get_data(self):
        from sklearn.datasets import make_blobs
        np.random.seed(self.random_state)
        if isinstance(self.n_samples, int):
            centers = self.n_centers
        else:
            if self.centers is None:
                # np.random.seed(self.random_state)
                centers = (np.random.normal(size=(self.n_centers, self.n_features)))
            else:
                centers = self.centers
        x, c = make_blobs(n_samples=self.n_samples, centers=centers, cluster_std=self.x_stds,
                          n_features=self.n_features, random_state=self.random_state)

        for k in range(self.n_centers):
            idxs = self.x_dependencies[k]
            for i in range(self.n_features):
                if i not in idxs:
                    n = len(x[c == k, i])
                    xmin = x[:, i].min()
                    xmax = x[:, i].max()
                    x[c == k, i] = np.random.randint(2, size=n)

        t, e, t_uncensored = self.get_te(x, c)
        return x, c, t, e, t_uncensored