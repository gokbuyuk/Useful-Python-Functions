# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
y_pred_test = model.predict_proba(X_test[features])[:, 1]
# fit a model
# predict probabilities

pred_test = (model.predict_proba(X_test[features])[:,1] >= thresholdOpt).astype(bool)

# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
auc = roc_auc_score(y_test, y_pred_test)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Model: ROC AUC=%.3f' % (auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred_test)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Model')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.title('Model')
# plt.savefig('Model_roc.png', bbox_inches='tight')
plt.show()

# Calculate ecdf
sns.ecdfplot(data=df_ml_quarters, x='Probability_1', hue='Disposition')
plt.title('''ANE: Cumulative distribution of assigned probabilities
          by Disposition - Model 1''')
plt.axvline(x = thresholdOpt, color = 'r', label = 'Ths')
plt.savefig('ANE_Model1_probs_ecdf.png', bbox_inches='tight')
plt.show()

# Feature importance using dalex
import dalex
exp = dx.Explainer(clf, X_train[features], y_train, label = "")
vi = exp.model_parts(loss_function='1-auc')
order_features_model1 = vi.result.sort_values(by='dropout_loss', ascending=False).head(60)

feature_names = []
feature_corr = []
feature_pvalues = []
for i in range(1,len(order_features_model1)):
    col = order_features_model1.iloc[i,0]
    if col not in ['_baseline_', '_full_model_']:
        feature_names.append(col)
        a = df[col]
        b = df[target]
        corr, pvalue = stats.pearsonr(a, b)
        feature_corr.append(corr)
        feature_pvalues.append(pvalue)
    
    
t = pd.DataFrame(list(zip(feature_names,ane_corr,ane_pvalues)), columns=['Feature','Correlation','p-value']) #.to_csv('TX_ANE_correlation.csv')

t['Direction'] = np.where(t['Correlation']<0, 'Negative', 'Positive')
# t.loc[:, 'Correlation'] = np.abs(t['Correlation'])
order_features_model1.rename(columns={'variable': 'Feature'}, inplace=True)
order_features_model1 = pd.merge(order_features_model1, t[['Feature', 'Direction']], on='Feature', 
                      how='left')
order_features_model1['Direction'].fillna('', inplace=True)
order_features_model1.loc[:, 'Feature'] = order_features_model1['Feature'].str.replace('_', ' ').str.capitalize()
t_sorted = order_features_model1.sort_values(by='dropout_loss', ascending=False)
fig,ax = plt.subplots(figsize=(5,7))
sns.barplot(x='dropout_loss', y='Feature', hue='Direction', hue_order=['Positive', 'Negative', ''], data=t_sorted, ax=ax, dodge=False)
ax.set_title('Drop-out Loss of the Model')
ax.set_ylabel('Feature')
# order_features_model1
