# Credit: Tien Pham
# Correlation detector
def detect_corr(data, target, corr_ths=0.9):
    '''
    Checks pairwise correlation, return if it is higher than corr_ths (correlation threshold).
    Issues an assert error in case such a variable pair is found.
    Prints the correlations of both predictor variables with the
    target variable so that can decide to exclude the variable
    that shows _less_ correlation with the target variable.
    '''
    # assert isinstance(data,pd.DataFrame)
    nc = len(data.columns)
    cn = data.columns
    for i in range(nc-1):
        x1 = data[cn[i]]
        for j in range(i+1,nc):
            x2 = data[cn[j]]
#             print(x1,x2)
            corr, _ = pearsonr(x1, x2)
            if abs(corr) > corr_ths:
                corr1 = 0
                corr2 = 0
                if outcome is not None:
                    corr1, _ = pearsonr(target, x1)
                    corr2, _ = pearsonr(target, x2)
                print("High correlation:", corr, cn[i], cn[j], corr1, corr2 )
