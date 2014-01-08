'''
Predicting the availability of parking spaces in Ljubljana car parks.
Luis Rei, me@luisrei.com, http://luisrei.com, @lmrei

More information at:
https://github.com/lrei/carpark_prediction

Notes: 
This was done as part of a course assignment for the Josef Stefan International
Postgraduate School.

This is mostly an ugly version of a iPython Notebook since I was using tmux with
vim and ipython panes... the code is uncommented and ugly but it is complementary
to both the report and the presentation and might be useful if you plan on doing
something similar.

If I ever need to do this again, I will clean it up.

The MIT License (MIT)

Copyright (c) 2014 Luis Rei

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

'''

import os
import gzip
import json
import unicodedata
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.api as tsa
from collections import defaultdict
from pandas import Series, DataFrame
from operator import itemgetter
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


INTERVALS = ['30Min', '60Min', '120Min', '180Min']


def read_files(data_dir='./data'):
    '''
    Read json files into a list.

    Args:
        data_dir: the directory containing the data files.

    Returns:
        One big list of entries dictionaries
    '''

    content = []
    for file in os.listdir(data_dir):
        path = os.path.join(data_dir, file)
        envelopes = json.loads(gzip.open(path, 'rb').read())
        for envelope in envelopes:
            json_data = json.loads(envelope['json_data'])
            for pk in json_data['Parkirisca']:
                park = {}  # new dictionary object
                # adds envole 'updated' field
                park['updated'] = int(json_data['updated'])
                # translations
                park['name'] = pk['Ime']
                park['price'] = pk['Cena_ura_Eur']

                park['park_id'] = pk['ID_ParkiriscaNC']
                if 'KoordinataX_wgs' in pk:
                    park['coordinate_x'] = pk['KoordinataX_wgs']
                    park['coordinate_y'] = pk['KoordinataY_wgs']
                else:
                    park['coordinate_x'] = None
                    park['coordinate_y'] = None
                if 'zasedenost' in pk:
                    park['occupied_places'] = \
                        int(pk['zasedenost']['P_kratkotrajniki'])
                    park['timestamp'] = \
                        int(pk['zasedenost']['Cas_timestamp'])
                else:
                    park['occupied_places'] = None
                    park['timestamp'] = None
                # unused
                #park['total_places'] = pk['St_mest']
                #park['workday_string'] = pk['U_delovnik']
                # add entry
                content.append(park)

    return content


def cleanup(content, remove_useless=True, remove_empty=False, sort_data=False):
    '''Perform based cleanup and ordering.'''

    # remove parks with no occupancy data at all
    if remove_useless:
        haves = set([x['park_id'] for x in content if 'occupied_places' in x])
        have_nots = set([x['park_id'] for x in content
                        if 'occupied_places' not in x])
        never_haves = have_nots.difference(haves)
        content = [x for x in content if x['park_id'] not in never_haves]

    # remove no occupancy entries
    if remove_empty:
        content = [x for x in content if 'occupied_places' in x]

    # sort by time updated
    if sort_data:
        content = sorted(content, key=itemgetter('updated'))

    return content


def import_into_sqlite(data_dir='./data', db='parking.db'):
    '''
    Read json files into a sqlite.

    Args:
        data_dir: the directory containing the data files.

    Returns:
        One big list of entries dictionaries
    '''
    conn = sqlite3.connect(db)
    c = conn.cursor()
    # create table
    c.execute('''CREATE TABLE parking
                 (updated INTEGER, ts INTEGER, park_id INTEGER, name TEXT,
                  price REAL, coordinate_x REAL, coordinate_y REAL,
                  total_places INTEGER, occupied_places INTEGER)''')

    for file in os.listdir(data_dir):
        path = os.path.join(data_dir, file)
        envelopes = json.loads(gzip.open(path, 'rb').read())
        for envelope in envelopes:
            json_data = json.loads(envelope['json_data'])
            for pk in json_data['Parkirisca']:
                park = {}  # new dictionary object
                # adds envole 'updated' field
                park['updated'] = str(int(json_data['updated']))
                # translations
                park['name'] = pk['Ime']
                park['price'] = pk['Cena_ura_Eur']
                park['total_places'] = pk['St_mest']
                park['park_id'] = pk['ID_ParkiriscaNC']
                if 'KoordinataX_wgs' in pk:
                    park['coordinate_x'] = pk['KoordinataX_wgs']
                    park['coordinate_y'] = pk['KoordinataY_wgs']
                else:
                    park['coordinate_x'] = None
                    park['coordinate_y'] = None
                if 'zasedenost' in pk:
                    park['occupied_places'] = \
                        pk['zasedenost']['P_kratkotrajniki']
                    park['timestamp'] = \
                        str(int(pk['zasedenost']['Cas_timestamp']))
                else:
                    park['occupied_places'] = None
                    park['timestamp'] = None
                # unused
                #park['workday_string'] = pk['U_delovnik']
                # add entry
                t = (park['updated'], park['timestamp'], park['park_id'],
                     park['name'], park['price'], park['coordinate_x'],
                     park['coordinate_y'], park['total_places'],
                     park['occupied_places'])

                c.execute('''INSERT INTO parking VALUES (?,?,?,?,?,?,?,?,?)''',
                          t)
    conn.commit()
    #delete from parking where park_id in (-17, 5, 22, 23, 24, 25,26,33,38,72);
    conn.close()


def import_min_into_sqlite(data_dir='./data', db='parking.min.db'):
    '''
    Read json files into a sqlite.

    Args:
        data_dir: the directory containing the data files.

    Returns:
        One big list of entries dictionaries
    '''
    conn = sqlite3.connect(db)
    c = conn.cursor()
    # create table
    c.execute('''CREATE TABLE parking
                 (updated INTEGER, park_id INTEGER, name TEXT,
                  price REAL, free_places INTEGER)''')

    for file in os.listdir(data_dir):
        path = os.path.join(data_dir, file)
        envelopes = json.loads(gzip.open(path, 'rb').read())
        for envelope in envelopes:
            json_data = json.loads(envelope['json_data'])
            for pk in json_data['Parkirisca']:
                park = {}  # new dictionary object
                # adds envole 'updated' field
                park['updated'] = str(int(json_data['updated']))
                # translations
                park['name'] = pk['Ime']
                park['price'] = pk['Cena_ura_Eur']
                park['total_places'] = pk['St_mest']
                park['park_id'] = pk['ID_ParkiriscaNC']
                if 'zasedenost' in pk:
                    park['free_places'] = \
                        pk['zasedenost']['P_kratkotrajniki']
                    park['timestamp'] = \
                        str(int(pk['zasedenost']['Cas_timestamp']))
                else:
                    continue
                # unused
                #park['workday_string'] = pk['U_delovnik']
                # add entry
                t = (park['updated'], park['park_id'], park['name'],
                     park['price'], park['occupied_places'])

                c.execute('''INSERT INTO parking VALUES (?,?,?,?,?)''', t)
    conn.commit()

    # delete parks without free_places data
    c.execute('''delete from parking where park_id in '''
              '''(-17, 5, 22, 23, 24, 25, 26, 33, 38, 72)''')
    # delete problematic parks (18 - two different) and parks with little data
    c.execute('''delete from parking where park_id in '''
              '''(7, 14, 18, 19, 20, 21)''')

    # delete obvious errors
    c.execute('''delete from parking where free_places < 0''')
    c.execute('''update parking_min set free_places = NULL '''
              '''where free_places > 2500''')
    conn.commit()

    # create table
    c.execute('''CREATE TABLE parking_min '''
              '''(updated INTEGER, park_id INTEGER, name TEXT,  price REAL, '''
              '''free_places INTEGER, PRIMARY KEY (park_id, updated) )''')
    c.execute('''INSERT OR IGNORE INTO parking_min SELECT updated, park_id,
                 name, price, free_places from parking''')
    conn.commit()

    conn.close()


def create_df(db='parking.min.db', save_as='parking.df.pickle'):
    conn = sqlite3.connect(db)
    rows = conn.execute('''select updated, park_id, free_places
                        from parking_min''').fetchall()
    ids = list(set([t[1] for t in rows]))
    data = {}
    for x in ids:
        dates = [np.datetime64(r[0], 's')
                 for r in rows if r[1] == x]   # updated
        y = [r[2] for r in rows if r[1] == x]  # free_places (target)
        data[x] = Series(y, index=dates)

    # convert data to DataFrame
    df = DataFrame(data)
    # get the names
    nr = conn.execute('''SELECT DISTINCT name
                      FROM parking ORDER BY park_id''').fetchall()
    # replace non ascii chars
    names = [unicodedata.normalize('NFKD', x[0]).encode('ascii', 'ignore')
             for x in nr]
    # remove dots
    names = [x.replace(u'.', '') for x in names]
    # assign to columns
    df.columns = names

    # destroy where there all are NaNs
    df = df[pd.notnull(df).any(axis=1)]

    # save
    if save_as is not None:
        df.to_pickle(save_as)

    return df


def resample_df(original_df, rs_interval='60Min', rs_how='last',
                window_size=4):
    # resample
    df = original_df.copy()
    rs = original_df.resample(rs_interval, how=rs_how)
    df = DataFrame(rs)
    df = df[pd.notnull(df).any(axis=1)]  # remove pull NaN rows

    # add windows
    for k in df.keys():
        for ind in range(1, window_size):
            vn = unicode(k) + u'-' + unicode(ind)
            df[vn] = np.hstack((np.array([np.NaN] * ind),
                                df[k].values))[:-ind]

    # destroy first lines
    df = df[window_size - 1:]  # this -1 is destroyed later

    return df


def create_park_df(df, k, no_missing=True):
    kdf = df.copy()

    # disregard entries where target variable is null/na
    if no_missing:
        kdf = kdf[kdf[k].notnull()]

    # create target variable
    kdf['target'] = np.hstack((np.array([np.NaN]), kdf[k].values))[:-1]

    # destroy first line (no prediction)
    kdf = kdf[1:]

    return kdf


def create_csvs(original_df, rs_interval='60Min', rs_how='last',
                window_size=10, work_dir="work"):
    data_files = []

    df = resample_df(original_df, rs_interval, rs_how, window_size)

    # create additional date fields
    df['date_year'] = df.index.year
    df['date_month'] = df.index.month
    df['date_day_month'] = df.index.day
    df['date_day_week'] = df.index.weekday
    df['date_hour'] = df.index.hour
    df['date_min'] = df.index.minute

    for k in original_df.keys():
        kdf = create_park_df(df, k)
        # write to csv
        filename = str(k) + '.csv'
        filepath = os.path.join(work_dir, filename)
        kdf.to_csv(filepath, index=False, index_label=False)
        data_files.append(filepath)

    return data_files


def split_df(df, delta=np.timedelta64(3, 'M')):
    # split into train - test
    last = df.index[-1] - delta

    train = df[df.index < last]
    test = df[df.index > last]

    return (train, test)


def split_dataset(df):
    train = df[df.is_train == True]
    test = df[df.is_train == False]

    return (train, test)


def load_datasets(data_file):
    '''Takes a park CSV file and returns the train and test set'''
    # read data file
    df = pd.read_csv(data_file)

    # load train and test set
    return split_dataset(df)


def load_all_testsets(work_dir='work'):
    '''not working correctly'''
    testsets = defaultdict(lambda: defaultdict(dict))
    for intv_dir in os.listdir(work_dir):
        if not intv_dir.endswith('Min'):
            continue
        intv_path = os.path.join(work_dir, intv_dir)
        for park_file in os.listdir(intv_path):
            if not park_file.endswith('.csv'):
                continue
            file_path = os.path.join(intv_path, park_file)
            park = park_file.split('.')[0]
            df = pd.io.parsers.read_csv(file_path)
            test = df[df.is_train == False]
            testsets[park][intv_dir] = test
    return testsets


def baseline_mean(train, test):
    preds = np.array([np.mean(train['target'].values)]
                     * len(test['target'].values))
    preds = np.round(preds)
    rmse = np.sqrt(mean_squared_error(test['target'].values, preds))
    return (preds, rmse)


def baseline_prev(test, basename):
    preds = test[basename].values
    preds = np.round(preds)
    rmse = np.sqrt(mean_squared_error(test['target'].values, preds))
    return (preds, rmse)


def linear_regression(train, test, features):
    regr = Pipeline([("imputer", Imputer(missing_values="NaN",
                                         strategy="mean",
                                         axis=0)),
                     ("linear", linear_model.LinearRegression())])
    regr.fit(train[features].values, train['target'].values)
    preds = regr.predict(test[features].values)
    preds = np.round(preds)
    rmse = np.sqrt(mean_squared_error(test['target'].values, preds))
    return (preds, rmse, regr)


def dt_regression(train, test, features):
    regr = Pipeline([("imputer", Imputer(missing_values="NaN",
                                         strategy="mean",
                                         axis=0)),
                     ("tree", DecisionTreeRegressor(max_depth=None,
                                                    min_samples_split=2,
                                                    min_samples_leaf=1))])
    regr.fit(train[features].values, train['target'].values)
    preds = regr.predict(test[features].values)
    preds = np.round(preds)
    rmse = np.sqrt(mean_squared_error(test['target'].values, preds))
    return (preds, rmse, regr)


def rf_regression(train, test, features):
    regr = Pipeline([("imputer", Imputer(missing_values="NaN",
                                         strategy="mean",
                                         axis=0)),
                    ("forest", RandomForestRegressor(n_estimators=20,
                                                     criterion='mse',
                                                     max_depth=None,
                                                     min_samples_split=2,
                                                     min_samples_leaf=1,
                                                     max_features='auto',
                                                     bootstrap=True,
                                                     oob_score=False,
                                                     n_jobs=-1,
                                                     random_state=None,
                                                     verbose=0,
                                                     min_density=None,
                                                     compute_importances=None))
                     ])
    regr.fit(train[features].values, train['target'].values)
    preds = regr.predict(test[features].values)
    preds = np.round(preds)
    rmse = np.sqrt(mean_squared_error(test['target'].values, preds))
    return (preds, rmse, regr)


def inc_linear_regression(train, test, features):
    test_size = len(test.values)

    # train - pred cycle
    preds = np.array([])
    # loop for each test case
    for ii in range(0, test_size):
        # build train and test sets
        tmp_train = pd.concat((train, test[0:ii]))[ii:]  # limit to test_size
        tmp_test = test[ii:ii + 1]  # just the next 1

        # Linear Regression
        (pred, _) = linear_regression(tmp_train, tmp_test, features)
        preds = np.hstack((preds, pred))

    # Calculate Scores
    rmse = np.sqrt(mean_squared_error(test['target'].values, preds))
    return (preds, rmse)


def features_for_park(df, basename):
    features_global = df.columns - ['target']
    features_local = [x for x in features_global if x.startswith(basename)]
    return features_local


def learn_test(train, test, basename):
    test_size = len(test.values)

    # feature groups
    features_global = train.columns - ['target']
    features_local = [x for x in features_global if x.startswith(basename)]
    #features_date = [x for x in features_global if x.startswith('date_')]
    #features_auto = basename

    print "\n"
    print "Park: %s" % (basename,)
    print "Train size: %d" % (len(train.values),)
    print "Test size: %d" % (test_size,)
    print

    # Result Dataframe
    results = DataFrame()
    results['target'] = test['target'].values

    # baseline: training set mean
    (preds_m, rmse_m) = baseline_mean(train, test)
    print "Baseline Mean: %f" % (rmse_m,)

    # baseline 2: previous value
    (preds_p, rmse_p) = baseline_prev(test, basename)
    print "Prev Val: %f" % (rmse_p,)

    # Model 1: Linear Regression - Local
    (preds_lr, rmse_lr, _) = linear_regression(train, test, features_local)
    print "Linear Regression: %f" % (rmse_lr)

    # Model 2: Random Forest  Regression - Local
    (preds_dt, rmse_dt, _) = dt_regression(train, test, features_local)
    print "Decision Tree Regression: %f" % (rmse_dt)

    # Model 3: Random Forest  Regression - Local
    (preds_rf, rmse_rf, _) = rf_regression(train, test, features_local)
    print "Random Forest Regression: %f" % (rmse_rf)

    #(preds_lri, rmse_lri) = inc_linear_regression(train, test, features_local)
    #print "Incremental Linear Regression: %f" % (rmse_lri,)
    #results['inc_linear_regression'] = preds_lri

    #build result dataframe, save to file
    results['baseline_mean'] = preds_m
    results['baseline_prev'] = preds_p
    results['linear_regression'] = preds_lr
    results['decision_tree'] = preds_dt
    results['random_forest'] = preds_rf

    return results


def learn_test_save(data_file, result_dir='results', keep_test=False):
    basename = os.path.split(data_file)[-1].split('.')[0]
    (train, test) = load_datasets(data_file)

    results = learn_test(train, test, basename)
    filename = os.path.split(data_file)[-1]
    filepath = os.path.join(result_dir, filename)
    results.to_csv(filepath, index=False, index_label=False)

    if keep_test:
        df = test.copy()
        # @TODO @HACK 1: features should not come from here
        for k in df.keys() - ['target']:
            if not k.startswith(basename):
                df = df.drop(k, axis=1)
        for k in results.keys() - ['target']:
            df[k] = results[k].values
        df.to_csv(filepath, index=False, index_label=False)


def load_results(result_dir='results'):
    results = defaultdict(lambda: defaultdict(dict))
    for intv_dir in os.listdir(result_dir):
        if not intv_dir.endswith('Min'):
            continue
        intv_path = os.path.join(result_dir, intv_dir)
        for park_file in os.listdir(intv_path):
            if not park_file.endswith('.csv'):
                continue
            file_path = os.path.join(intv_path, park_file)
            park = park_file.split('.')[0]
            results[park][intv_dir] = pd.io.parsers.read_csv(file_path)
    return results


def calc_rmse(df, methods=None):
    rmse = {}
    targets = df['target'].values
    if methods is None:
        methods = df.keys() - ['target']

    for method in methods - ['target']:
        preds = df[method].values
        rmse[method] = np.sqrt(mean_squared_error(targets, preds))
    return rmse


def calc_rmse_global(results, methods=None):
    ''' can be converted to a pd.DataFrame for better display'''
    rmse_global = defaultdict(lambda: defaultdict(list))
    for park in results:
        for interval in results[park]:
            rmse = calc_rmse(results[park][interval], methods)
            for method in rmse:
                rmse_global[interval][method].append(rmse[method])
    for interval in rmse_global:
        for method in rmse_global[interval]:
            rmse_global[interval][method] = \
                np.average(rmse_global[interval][method])
    return rmse_global


def total_rmse(rmse):
    total = defaultdict(int)
    for intv in rmse:
        for method in rmse[intv]:
            total[method] += rmse[intv][method]
    return total


def plot_summary(results, interval='30Min', graphdir='graphs'):
    fig, ax = plt.subplots()
    n_groups = len(results.keys())
    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 0.4

    summary = defaultdict(list)
    for park in sorted(results.keys()):
        rmse = calc_rmse(results[park][interval])
        for method in rmse:
            summary[method].append(rmse[method])

    pos = 0
    colors = ['b', 'c', 'k', 'r']
    for keyname in sorted(summary.keys()):
        plt.bar(index + bar_width * pos, summary[keyname], bar_width,
                alpha=opacity, color=colors[pos], label=keyname)
        pos += 1

    plt.xlabel('Park')
    plt.ylabel('RMSE')
    plt.title('RMSE for each Park - ' + interval + ' intervals')

    names = sorted(results.keys())
    names = [x[0:8] for x in names]
    plt.xticks(index + bar_width, tuple(names))
    plt.legend()

    #plt.tight_layout()
    plt.show()
    #filepath = os.path.join(graphdir, 'summary' + '-' + interval + '.png')
    #plt.savefig(filepath, dpi=fig.dpi)


def plot_park(df, date_start, date_end, freq='2H'):
    # set the major xticks and labels through pandas
    ax2 = plt.figure().add_subplot(111)
    xticks = pd.date_range(start=date_start, end=date_end, freq=freq)
    df.plot(ax=ax2, style='-v', label='second line',
            xticks=xticks.to_pydatetime())
    ax2.set_xticklabels([x.strftime('%a\n%H\n%M\n') for x in xticks])
    ax2.set_xticklabels([], minor=True)
    # turn the minor ticks created by pandas.plot off
    # plt.minorticks_off()
    plt.show()


def window_test(df, intervals, windows):
    test_results = defaultdict(list)
    for window_size in windows:
        print window_size
        for intv in intervals:
            rdf = resample_df(df, intv, 'last', window_size)
            for park in df.keys():
                parkdf = create_park_df(rdf, park)
                features = features_for_park(parkdf, park)
                train, test = split_df(parkdf)
                preds, rmse = rf_regression(train, test, features)
                test_results[window_size].append(rmse)

    for key in test_results:
        test_results[key] = np.average(test_results[key])

    values = np.array(test_results.values())
    index = np.array(test_results.keys())

    result_df = pd.DataFrame(values, index=index)
    result_df.columns = ['RMSE (avg)']

    return result_df


def autoregressive(original_df, intervals=INTERVALS, rs_how='last'):
    ''' unfinished '''
    for intv in intervals:
        kdf = original_df.resample(intv, how=rs_how)
        for k in kdf.keys():
            kdf = kdf[k]
            kdf = kdf[kdf.notnull()]
            train, test = split_df(kdf)
            arma = tsa.ARMA(train.values, order=(2, 2))
            model = arma.fit()

            #preds = model.predict(start=test.index[0], end=test.index[-1])
            preds = model.predict(0, len(test.values))
            rmse = np.sqrt(mean_squared_error(test.values, preds))
            print rmse


def main():
    intervals = INTERVALS
    base_work_dir = 'work'
    base_result_dir = 'results'
    dataframe_file = 'parking.pickle'

    if os.path.exists(dataframe_file):
        print 'Loading Base DataFrame'
        df = pd.io.pickle.read_pickle(dataframe_file)
    else:
        print 'Creating Base DataFrame'
        df = create_df(save_as=dataframe_file)

    for intv in intervals:
        intv_result_dir = os.path.join(base_result_dir, intv)

        intv_work_dir = os.path.join(base_work_dir, intv)
        if not os.path.exists(intv_work_dir):
            print 'Creating CSV files for %s' % (intv,)
            os.makedirs(intv_work_dir)
            data_files = create_csvs(df, rs_interval=intv, rs_how='last',
                                     window_size=4, work_dir=intv_work_dir)
        else:
            data_files = os.listdir(intv_work_dir)
            data_files = [os.path.join(intv_work_dir, x) for x in data_files]

        if not os.path.exists(intv_result_dir):
            print 'Learn-Test for %s (%d parks)' % (intv, len(data_files))
            os.makedirs(intv_result_dir)
            for data_file in data_files:
                learn_test_save(data_file, intv_result_dir)

    results = load_results()
    rmse = calc_rmse_global(results)
    resdf = DataFrame(rmse)
    print resdf


if __name__ == '__main__':
    main()
