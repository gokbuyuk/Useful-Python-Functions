import dalex as dx
# https://dalex.drwhy.ai/python/api/#dalex.Explainer.model_parts
features = []
exp = dx.Explainer(model, X_train[features], y_train)
vi = exp.model_parts(loss_function='1-auc')
order_features_model1 = vi.result.sort_values(by='dropout_loss', ascending=False)

feature_imp = order_features_model1[order_features_model1['variable'].isin(features)]
for feature in features:
    temp_df = Xy_train[[feature, target]].dropna()
    r, p = stats.pearsonr(temp_df[feature], temp_df[target])
    feature_imp.loc[feature_imp['variable']==feature, 'Direction'] = np.where( r< 0,  
    'Negative', 'Positive')
    
sns.barplot(data=feature_imp, y='variable', x='dropout_loss', hue='Direction', dodge=False)

