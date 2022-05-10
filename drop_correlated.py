#### Drop features that are highly correlated with others
# Create correlation matrix
corr_matrix = X.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than ths
ths = 0.95
to_drop = [column for column in upper.columns if any(upper[column] > ths)]
# to_drop
print('These features are dropped:')
print(to_drop)
# # Drop features 
X.drop(to_drop, axis=1, inplace=True)
