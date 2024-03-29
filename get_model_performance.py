model_metrics = pd.read_csv('model_metrics.csv', header=0)

model_metrics = pd.DataFrame(columns=['Accuracy', 
                          'FPR','FNR', 'Precision', 'Recall', 'Auc_train', 'Auc_val', 'Auc_test', 'n_bucket',
                          'Q1-Positive Rate', 'Qn-Positive Rate', 'n_features', 'Features', 'Feature Importances', 'Model'])

scale = y_train.value_counts()[0]/y_train.value_counts()[1]

model = xgb.XGBClassifier(scale_pos_weight = np.sqrt(scale),
                         random_state = 42,
                         eval_metric = 'auc',
                         use_label_encoder=False,
                         colsample_bytree = 0.6910552119217912,
                        gamma = 1.0890531809815442,
                        max_depth = 6,
                        min_child_weight = 5,
                        reg_alpha = 42,
                        reg_lambda = 0.11872125743598791
                         )

def get_model_performance(model, target, features):
  ''' Get the performance of the binary classification model 
  using the given list of features on training, validation 
  and test sets 
  model: classifier,
  features: list of column names to train the model with
  target: str target column name
  '''
  clf = model.fit(X_train[features],y_train)
  n_bucket = 4

  test_probs_0 = model.predict_proba(X_train[features])[:, 0]
  quarters = pd.qcut(test_probs_0, 4, labels=list(range(1,n_bucket+1)))
 
  probs = pd.DataFrame({'Prob_0' :test_probs_0, 
                          'Quarter' :quarters})
  cutoffs = probs.groupby('Quarter').max().reset_index()
  cutoff_probs = [-0.1] + list(cutoffs['Prob_0'])[:-1] + [1.1] 

  df_ml_probs_0 = model.predict_proba(X_val[features])[:, 0]
  prob_1 = [1 - x for x in df_ml_probs_0]
  quarters = pd.cut(df_ml_probs_0, cutoff_probs, labels=list(range(1,n_bucket+1)), duplicates='raise')
  df_ml_quarters = pd.DataFrame({'Probability_1': prob_1,
                                'Quarter': quarters,
                                target: y_val})
  perf = df_ml_quarters.groupby('Quarter')[target].mean().round(3).reset_index()

  # y_quarter1 = perf[perf['Quarter']==1][target].mean()
  # y_quarter4 = df_ml_quarters[df_ml_quarters['Quarter']==4][target].mean()
  Q1_Positive_rate = perf[target][0]
  Qn_Positive_rate = perf[target][n_bucket-1]


  y_pred = model.predict_proba(X_train[features])[:,1]
  fpr, tpr, thresholds = roc_curve(y_train, y_pred)
  gmean = np.sqrt(tpr * (1 - fpr))
  index = np.argmax(gmean)
  thresholdOpt = round(thresholds[index], ndigits = 4)
  pred = (model.predict_proba(X_val[features])[:,1] >= thresholdOpt).astype(bool)

  tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
  sum = tn+fp+fn+tp
  accuracy = round((tp+tn)/sum,3)
  tpr = round((tp/(tp+fn)),3)
  fpr = round(fp/(fp+tn),3)
  precision = round(tp/(tp+fp),3)
  recall = round((tp/(tp+fn)),3)
  fnr = round(fn/(fn+tp),3)
  auc_train = round(roc_auc_score(y_train, clf.predict_proba(X_train[features])[:,1]),3)
  auc_val = round(roc_auc_score(y_val, clf.predict_proba(X_val[features])[:,1]),3)
  auc_test = round(roc_auc_score(y_test, clf.predict_proba(X_test[features])[:,1]),3)
  # mcc = round(metrics.matthews_corrcoef(y_val, pred),3)
  # f1 = round(f1_score(y_val, pred, average='binary'),3)
              
  new_row = {'Accuracy': accuracy,
              'FPR': fpr, 
              'FNR': fnr,
              'Precision': precision,
              'Recall': recall,
              'Auc_train': auc_train,
              'Auc_val': auc_val, 
              'Auc_test': auc_test,
              'n_bucket': n_bucket,
              'Q1-Positive Rate': Q1_Positive_rate,
              'Qn-Positive Rate': Qn_Positive_rate,  
              'n_features': len(features),
              'Features': list(features),
              'Feature Importances': dict(zip(features, model.feature_importances_)),
              'Model': model.get_xgb_params()
   }
  return(new_row)

new_row = get_model_performance(model, features)
model_metrics = model_metrics.append(new_row, ignore_index=True)
model_metrics.to_csv('model_metrics.csv', index=False)
model_metrics.tail()
