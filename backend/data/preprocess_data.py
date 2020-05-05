# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def preprocess_data(file, tag):
    if tag == 'Amazon':
        df = pd.read_csv(file)
        df.iloc[:, 0] = 'amazon'
        df.drop(['Time'], axis=1, inplace=True)
        df.insert(2, 'Category', df['Title']) 

    elif tag == 'Google':
        df = pd.read_csv(file)
        df = df.rename(columns={'Minimum Qualifications': 'Minimum_Qualifications', 
                                'Preferred Qualifications': 'Preferred_Qualifications'})
    
    elif tag == 'NYC':
        df = pd.read_csv(file)
        deleted_cols = ['Job ID', 'Posting Type', '# Of Positions', 'Title Code No', 
                        'Level', 'Job Category', 'Full-Time/Part-Time indicator', 
                        'Salary Range From', 'Salary Range To', 'Salary Frequency', 
                        'Division/Work Unit', 'Additional Information', 'To Apply', 
                        'Hours/Shift', 'Work Location 1', 'Recruitment Contact', 
                        'Residency Requirement', 'Posting Date','Post Until', 
                        'Posting Updated', 'Process Date']
        df.drop(columns=deleted_cols, inplace=True)
        df.columns=['Company', 'Title', 'Category', 'Location', 'Responsibilities', 
            'Minimum_Qualifications', 'Preferred_Qualifications']
            
    df = df.dropna(how='any',axis='rows')
    df['Company'] = df.Company.apply(lambda x: x.lower())
    df['Company'] = df.Company.apply(lambda x: word_tokenize(x))

    df['Title'] = df.Title.apply(lambda x: x.lower())
    df['Title'] = df.Title.apply(lambda x: word_tokenize(x))
    df['Category'] = df.Category.apply(lambda x: x.lower())
    df['Category'] = df.Category.apply(lambda x: word_tokenize(x))
    df['Location'] = df.Location.apply(lambda x: x.lower())
    df['Location'] = df.Location.apply(lambda x: word_tokenize(x))

    df['Responsibilities'] = df.Responsibilities.apply(lambda x: x.lower())
    df['Responsibilities'] = df.Responsibilities.apply(lambda x: word_tokenize(x))
    df['Responsibilities'] = df.Responsibilities.apply(lambda x: [w for w in x if w not in stop_words])

    df['Minimum_Qualifications'] = df.Minimum_Qualifications.apply(lambda x: x.lower())
    df['Minimum_Qualifications'] = df.Minimum_Qualifications.apply(lambda x: word_tokenize(x))
    df['Minimum_Qualifications'] = df.Minimum_Qualifications.apply(lambda x: [w for w in x if w not in stop_words])

    df['Preferred_Qualifications'] = df.Preferred_Qualifications.apply(lambda x: x.lower())
    df['Preferred_Qualifications'] = df.Preferred_Qualifications.apply(lambda x: word_tokenize(x))
    df['Preferred_Qualifications'] = df.Preferred_Qualifications.apply(lambda x: [w for w in x if w not in stop_words])

    return df

def main():
  df_a = preprocess_data('amazon_jobs_dataset.csv', tag='Amazon')
  df_g = preprocess_data('job_skills.csv', tag='Google')
  df_n = preprocess_data('nyc-jobs.csv', tag='NYC')
  df = pd.concat([df_a, df_g, df_n])
  df.to_csv('data.csv', index=False)

if __name__ == '__main__':
  main()
