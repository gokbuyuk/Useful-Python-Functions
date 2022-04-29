def merge_rows(data, duplicatedcols, colstochange=None, func=None, keep='first', sortedby=None, ascending=True):
    '''Removes duplicates from given data frame if the rows have the same values in duplicatedcols,
    applies the given fcn on colstochange list of columns, 
    returns the dataframe with these changed made.
    
    data: pandas DataFrame,
    colstochange: list of columns names to modify during merging, 
                if there are multiple lists of columns to be treated differently, 
                provide a list of functions for each sublist of cloumns in the same order
    fcn: function to modify to the colstochange or a list of functions
    sortedby: name of column to sort by
    ascending: {True, False}, default False
    keep: {'first', 'last', False}, default 'first'
    '''
    if func==list:
        rows_merged = []
        for i in range(len(func)):
            # func_i = func[i]
            colstochange_i = colstochange[i]
            rows_merged[i] = data.groupby(duplicatedcols).agg({col: func for col in colstochange_i})
            data = data.sort_values(by=sortedby, ascending=ascending).drop(colstochange_i, axis=1).drop_duplicates(duplicatedcols, keep=keep)
        
        for i in range(len(rows_merged)):
            data = data.merge(rows_merged[i], how='left', on=duplicatedcols) 
    
    else:
        rows_merged = data.groupby(duplicatedcols).agg({col: func for col in colstochange})
        data = data.sort_values(by=sortedby, ascending=ascending).drop(colstochange, axis=1).drop_duplicates(duplicatedcols, keep=keep)
        data = data.merge(rows_merged, how='left', on=duplicatedcols)
                   
    
    return(data)  

#####  example:df_cleaned = merge_rows(df, duplicatedcols=['col1', 'col2'],
#         colstochange=['col3', 'col4', 'col5'],
#         func='max', keep='first', sortedby='Key ID', ascending=True )
