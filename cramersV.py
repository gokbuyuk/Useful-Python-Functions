def cramersV(df, target):
  '''Find the Cramer's V between categorical variables with a target variable'''
    output = pd.DataFrame(columns=['Feature', "Cramers V"])
    for col in df.columns.difference([target]).tolist():
        data_crosstab = pd.crosstab(df[col], 
                                    df[target],
                                        margins = False)
        data_crosstab = data_crosstab.to_numpy()
        #Chi-squared test statistic, sample size, and minimum of rows and columns
        X2 = stats.chi2_contingency(data_crosstab, correction=False)[0]
        n = np.sum(data_crosstab)
        minDim = min(data_crosstab.shape)-1
        #calculate Cramer's V 
        V = np.sqrt((X2/n) / minDim)
        output.loc[len(output.index)] = [col, V]
        
    return(output)
