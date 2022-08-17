import dalex as dx
# https://dalex.drwhy.ai/python/api/#dalex.Explainer.model_parts
features = []
exp = dx.Explainer(model, X_train[features], y_train)
vi = exp.model_parts(loss_function='1-auc')
dx_feature_imp = vi.result.sort_values(by='dropout_loss', ascending=False)
base_score = dx_feature_imp.loc[dx_feature_imp['variable'] == '_full_model_', 	'dropout_loss'].tolist()[0]
dx_feature_imp['dropout_loss'] = dx_feature_imp['dropout_loss'] - base_score 

dx_feature_imp = dx_feature_imp[dx_feature_imp['variable'].isin(features)]
for feature in features:
    temp_df = Xy_train[[feature, target]].dropna()
    r, p = stats.pearsonr(temp_df[feature], temp_df[target])
    dx_feature_imp.loc[dx_feature_imp['variable']==feature, 'Direction'] = np.where( r< 0,  
    'Negative', 'Positive')
    
sns.barplot(data=dx_feature_imp, y='variable', x='dropout_loss', hue='Direction', dodge=False)
plt.title('Dropout loss for features in the model')
plt.xlabel('1-AUC')
plt.ylabel('Feature')
plt.legend(loc='center right')
