risk_groups = []

def risk_score(data, target, model, risk_categories, risk_names):
  ''' Generates risk score based on the probability obtained using the given binary classification model.
  data: DataFrame
  target: str, column name for the target variable
  model: sklearn model to be used for calculating risk score
  risk_categories: list of categories that are list of features to be used to calculate risk score for that category
  risk_names: list of names of the categories
  '''
  for i,features in enumerate(risk_groups):
      model.fit(X_train[features], y_train)
      y_pred_logit = model.predict(X_test[features])
      auc_train = round(roc_auc_score(y_train, model.predict_proba(X_train[features])[:,1]),3)
      auc_test = round(roc_auc_score(y_test, model.predict_proba(X_test[features])[:,1]),3)
      print('AUC of logistic regression classifier on test set: {:.2f}'.format(auc_test))

      for df in [X_train, X_val, X_test]:
          prob_1 = model.predict_proba(df[features])[:, 1]
          df['Risk_score_{}'.format(i)] = prob_1

