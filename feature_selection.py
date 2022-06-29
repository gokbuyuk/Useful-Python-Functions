
metric_path = 'path_to_save.csv
N = 20
feature_scores_xgb = pd.DataFrame({'Feature': list(X_train.columns)})
for i in range(N):
  fs_xgb = xgb.XGBClassifier(scale_pos_weight = np.sqrt(scale),
                        #  random_state = 42,
                         max_depth=10,
                         eval_metric = 'auc',
                         use_label_encoder=False
                         )
  
  fs_xgb.fit(X_train, y_train)
  feature_scores_xgb['Score_'+ str(i)] = fs_xgb.feature_importances_
feature_scores_xgb['total_score'] = feature_scores_xgb.sum(axis=1)
feature_scores_xgb

features_xgbwinners = feature_scores_xgb.sort_values(by='total_score', ascending=False)['Feature'].tolist()


features = features_xgbwinners  
n_min = 25
n_max = min(50, len(features))
n_model = 5
step = (n_max-n_min)//n_model
n_feature = n_max
  
for m in range(n_model):
    new_row = get_model_performance(xgb0, features[:n_feature])
    model_metrics = model_metrics.append(new_row, ignore_index=True)
    
    # sort by the feature importance and update features with the top n_feature
    d = new_row['Feature Importances']
    d = dict(sorted(d.items(), key=lambda x: x[1], reverse=True)) 
    n_feature = n_feature - step
        
model_metrics.to_csv(metrics_path, index=False)
model_metrics.tail()

features_list = [features_xgbwinners]
for n in range(20,50,10):
   
    ## train n*m models using the intersection of top n features and building m models iteratively dropping the least important features
    for features in features_list:
              
        # Get the intersection of top n most important features
        for col in feature_scores_xgb.columns.difference(['Feature', 'total_score']).tolist():
            features =  list(set(feature_scores_xgb.sort_values(by=col, ascending=False).iloc[:n,]['Feature']).intersection(features))
        
        n_min = 10
        n_max = len(features)
        n_model = 4
        step = (n_max-n_min)//n_model
        n_feature = n_max
            
        for m in range(n_model):
            new_row = get_model_performance(xgb0, features[:n_feature])
            model_metrics = model_metrics.append(new_row, ignore_index=True)
            print(model_metrics.iloc[-1,:12])
            # sort by the feature importance and update features with the top n_feature
            d = new_row['Feature Importances']
            d = dict(sorted(d.items(), key=lambda x: x[1], reverse=True)) 
            n_feature = n_feature - step
            features = list(d.keys())[:n_feature]
        
model_metrics.to_csv(metrics_path, index=False)
model_metrics.tail()
