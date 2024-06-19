import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

languages_level_dict = {'A1': 1,
                        'A2': 2,
                        'B1' : 3,
                        'B2' : 4,
                        'C1': 5,
                        'C2': 6}




def process_language(df: pd.DataFrame, language: str, languages_level_dict: dict) -> pd.DataFrame:
    """
    Process the DataFrame to extract information about the specified language.

    Parameters:
    - df: pd.DataFrame
    - language: str, the language to process (e.g., 'German', 'English')
    - languages_level_dict: dict, a dictionary to replace the ratings

    Returns:
    - pd.DataFrame with additional columns for the specified language
    """
    
    df[language] = df['languages'].apply(lambda lst: next((d for d in lst if d.get('title') == language), 0))
    must_have_col = f'{language}_must_have'
    df[must_have_col] = df[language].apply(lambda x: 1 if (x != 0 and x.get('must_have') == True) else 0)
    df[language] = df[language].apply(lambda x: x.get('rating') if x != 0 else 0).replace(languages_level_dict)
    return df




def transfer(df:pd.DataFrame, with_lable=True, **kwargs) -> pd.DataFrame:
    df[["talent","job"]] = df[["talent","job"]].apply(dict)
    # prepare features for talant and job 
    talent_df = pd.json_normalize(df.talent)
    job_df = pd.json_normalize(df.job)
    
    #encode languages , there are only {'English', 'German'} in job desctiption,
    # must have put 1 if talent speaks this languege 

    talent_df = process_language(talent_df,"German",languages_level_dict) 
    talent_df['German_must_have'] = 1
    talent_df = process_language(talent_df,"English",languages_level_dict)
    talent_df['English_must_have'] = 1

    job_df = process_language(job_df,"German",languages_level_dict) 
    job_df = process_language(job_df,"English",languages_level_dict)


    # Encode job_role
    # determaine job_role_set from jobs description
    job_roles_list = list(set(job_df.job_roles.sum()))
    
    for role in job_roles_list:
        job_df[role] = job_df.job_roles.apply(lambda x: role in x)

    for role in job_roles_list:
        talent_df[role] = talent_df.job_roles.apply(lambda x: role in x)

    

    # Encode seniorities
    job_seniorities_list = list(set(job_df.seniorities.sum()))
    
    for seniority in job_seniorities_list:
        job_df[seniority] = job_df.seniorities.apply(lambda x: seniority in x)

    for seniority in job_seniorities_list:
        talent_df[seniority] = talent_df.seniority.apply(lambda x: seniority in x)

    # encode degree
    # Define degree level hierarchy
    degree_hierarchy = {
        'none': 0,
        'apprenticeship': 1,
        'bachelor': 2,
        'master': 3,
        'doctorate': 4
    }
    sorted_degrees = sorted(degree_hierarchy, key=lambda x: degree_hierarchy[x])
    
    for idx, degree in enumerate(sorted_degrees):
        if degree!='none':
            job_df[degree] = job_df.apply(
                lambda x: True if (x.min_degree in x) or (x[sorted_degrees[idx-1]]) else False, axis=1)
        else:       
            job_df[degree] = job_df.min_degree.apply(lambda x: degree in x)

    for degree in sorted_degrees:
        talent_df[degree] = talent_df.degree.apply(lambda x: degree in x)
        
 
    #add talent and job ids for model quality metrics calculation
    talent_df['talent_id'] = talent_df.index
    talent_df['talent_or_job'] = 'talent'
    talent_df['matched_job_id'] = df.label.values
    talent_df['matched_job_id'] = talent_df.apply(lambda x: x.name if x['matched_job_id'] else -1, axis=1)
    job_df['job_id'] = job_df.index
    job_df['talent_or_job']= 'job'
    job_df['matched_talent_id'] = df.label.values
    job_df['matched_talent_id'] = job_df.apply(lambda x: x.name if x['matched_talent_id']==True else -1,  axis=1)

    # unify salery column
    talent_df= talent_df.rename({"salary_expectation":"salary"}, axis=1)
    job_df= job_df.rename({"max_salary":"salary"}, axis=1)
    talent_df.drop(['languages','job_roles','seniority', 'degree'],axis=1, inplace=True)
    job_df.drop(['languages','job_roles','seniorities','min_degree'],axis=1, inplace=True)
    
    df = pd.concat([talent_df, job_df], ignore_index=True)
    df = df.fillna(-1)
    df = df.applymap(lambda x: int(x) if isinstance(x, bool) else x)

    df.to_csv('model_matrix.csv', index=False)
    # be used as feature storage
    scaler = StandardScaler()
    columns_not_to_scale = ['talent_id', 'talent_or_job', 'matched_job_id',
       'job_id', 'matched_talent_id']
    columns_to_scale = [col for col in df.columns if col not in columns_not_to_scale]
    scaled_columns = scaler.fit_transform(df[columns_to_scale])
    scaled_df = pd.DataFrame(scaled_columns, columns=columns_to_scale, index=df.index)
    df[columns_to_scale] = scaled_df
    
    # Save the fitted scaler to a file
    joblib.dump(scaler, 'scaler_selected_columns.pkl')

    # in order to speed up cosine similarity calculation,
    # normalise features by norm of the rows
    df[columns_to_scale] = np.divide(df[columns_to_scale], np.linalg.norm(df[columns_to_scale], axis=1, keepdims=True))
    df.to_parquet('model_matrix.pq', index=False)
    
    return df