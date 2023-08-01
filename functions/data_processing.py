import numpy as np
import pandas as pd

def data_processing(eto_file):
    '''
    transforming ETO data
    '''
    df = pd.read_csv('./ETO_data.csv')
    df.loc[:,'Medicaid'] = df.loc[:,'insurance'].apply(medicaid)
    df.loc[:,'private_insurance'] = df.loc[:,'insurance'].apply(private)
    df.loc[:,'obamacare'] = df.loc[:,'insurance'].apply(obamacare)
    
    process_weight_class(df)
    process_tobacco(df)
    
    
    #substance
    df.loc[~(df.loc[:,'alcohol_preg'].isna()),'alcohol_preg'] = df.loc[:,'alcohol_preg'].apply(sub_trans)
    df.loc[~(df.loc[:,'alcohol'].isna()),'alcohol'] = df.loc[:,'alcohol'].apply(sub_trans)
    df.loc[~(df.loc[:,'tobacco'].isna()),'tobacco'] = df.loc[:,'tobacco'].apply(sub_trans)
    
    df.loc[:,'condom'] = df.loc[:,'condom'].apply(condom_trans)
    df.loc[:,'birth_age'] =  pd.to_datetime(df.loc[:,'child_bday']).dt.year - pd.to_datetime(df.loc[:,'DOB_PP']).dt.year
    df.loc[df.loc[:,'birth_age']< 14,'birth_age'] = np.nan
    df.loc[:,'twin'] = df.loc[:,'singleton'].apply(recode_twin)
    df.loc[:,'enrollment_time'] = df.loc[:,'enrollment_time'].apply(enrol_time)
    df.loc[:,'first_prenatal_care_time'] = df.loc[:,'first_prenatal_care_time'].apply(care_time)
    df.loc[:,'edu'] = df.loc[:,'highest_edu'].apply(edu_trans)
    df.loc[:,'preventive'] = df.loc[:,'preventive'].apply(prev_trans)
    df.loc[:,'inc'] = df.loc[:,'household income'].apply(inc_trans) 
    df.loc[:,'inc_per_capita'] = np.log(df.loc[:,'inc'] / df.loc[:,'household size'])
    df.loc[:,'depression'] = df.loc[:,'depression'].apply(prev_trans)
    df.loc[:,'hypertension'] = df.loc[:,'hypertension'].apply(prev_trans)
    df.loc[:,'diabetes'] = df.loc[:,'diabetes'].apply(prev_trans)

    #previous pregnancy:'preterm', 'prev_LBW','prev_very_LBW','prev_hospitalized_birth'
    df.loc[df.loc[:,'preg_weeks']<37,'Preterm'] = 1
    df.loc[df.loc[:,'preg_weeks']>=37,'Preterm'] = 0    
    df.loc[:,'Prev_preterm'] = df.loc[:,'prev_preterm'].apply(prev_trans)
    df.loc[:,'Prev_LBW'] = df.loc[:,'prev_LBW'].apply(prev_trans)
    df.loc[:,'Prev_VLBW'] = df.loc[:,'prev_very_LBW'].apply(prev_trans)
    df.loc[:,'Prev_Hosp'] = df.loc[:,'prev_hospitalized_birth'].apply(prev_trans)
    #previous kid
    df.loc[:,'spacing_18m'] = df.loc[:,'prev_kid_age'].apply(spacing_18m)
    df.loc[:,'first_preg'] = df.loc[:,'prev_kid_age'].apply(first_preg)
    
    #adolescent or advanced pregnancy
    df.loc[df.loc[:,'birth_age'] < 19,'adol_preg'] = 1
    df.loc[df.loc[:,'birth_age'] >= 19,'adol_preg'] = 0
    df.loc[df.loc[:,'birth_age'] < 35,'adv_preg'] = 0
    df.loc[df.loc[:,'birth_age'] >= 35,'adv_preg'] = 1
    
    #correcting absolutely wrong numbers
    mask = df.loc[:,'household size'] > 10
    df.loc[mask,'household size'] = df.loc[mask,'household size'].apply(lambda x : x //10)
    
    df2 = df.loc[~(df.loc[:,'Sex'] == 'Male'),:]
    
    #previous adverse event
    df2.loc[(df.loc[:,'Prev_VLBW'] == 0) | (df.loc[:,'Prev_preterm'] == 0) | (df.loc[:,'Prev_LBW'] == 0) | (df.loc[:,'Prev_Hosp'] == 0),'Prev_ad'] = 0
    df2.loc[df.loc[:,'Prev_VLBW'] == 1, 'Prev_ad'] = 1
    df2.loc[df.loc[:,'Prev_preterm'] == 1, 'Prev_ad'] = 1
    df2.loc[df.loc[:,'Prev_LBW'] == 1, 'Prev_ad'] = 1
    df2.loc[df.loc[:,'Prev_Hosp'] == 1, 'Prev_ad'] = 1
    
    df2.to_csv('./eto_sample.csv')
    df.loc[:,'child_id'] = df.index
    
    rename_dict = {'SubjectID': 'parent_id','DOB_PP_x':'parent_bday'}
    df1 = df.rename(columns = rename_dict)
    
    lst = [#identification
    'child_id','parent_id','child_bday','parent_bday',
    #birth info
    'Preg_start','preg_weeks', 'Preterm','birth_weight_gram', 'weight_class',
    #program info and SES background
    'birth_age','edu','inc_per_capita','household size','preventive','enrollment_time',
    'first_prenatal_care_time','spacing_18m', 'first_preg', 'inc', 'adol_preg','adv_preg',
    #health background
    'mental health','tobacco','alcohol','condom', 'prev_preterm','Prev_preterm', 'Prev_LBW', 'Prev_VLBW',
    'Prev_Hosp','diabetes', 'hypertension', 'depression','Medicaid','private_insurance', 'obamacare',
    #during pregnancy
    'twin','alcohol_preg','tobacco_preg']
    
    df2 = df1.loc[:, lst]
    
    df2.to_csv('./cleaned_data/cleaned_bg.csv')

def process_tobacco(df):
     #tobacco
    df.loc[(df.loc[:,'cigarette_y'] != 'I don\'t smoke') & ~(df.loc[:,'cigarette_y'].isna()),'cigarette_pn'] = 1
    df.loc[(df.loc[:,'cigarette_y'] == 'I don\'t smoke'),'cigarette_pn'] = 0
    df.loc[(df.loc[:,'e-cigarette'] != 'Not at all') & (~(df.loc[:,'e-cigarette'].isna())),'e-cigarette'] = 1
    df.loc[(df.loc[:,'e-cigarette'] == 'Not at all'),'e-cigarette'] = 0
    df.loc[(df.loc[:, 'b. Hookah'] != 'Not at all') & (~(df.loc[:,'b. Hookah'].isna())), 'Hookah'] = 1
    df.loc[(df.loc[:, 'b. Hookah'] == 'Not at all'), 'Hookah'] = 0
    df.loc[(df.loc[:, 'd. Cigars, cigarillos, or little filtered cigars'] != 'Not at all')& (~(df.loc[:,'d. Cigars, cigarillos, or little filtered cigars'].isna())), 'd. Cigars, cigarillos, or little filtered cigars'] = 1
    df.loc[(df.loc[:, 'd. Cigars, cigarillos, or little filtered cigars'] == 'Not at all'), 'd. Cigars, cigarillos, or little filtered cigars'] = 0
    df.loc[(df.loc[:,'c. Chewing tobacco, snuff, snus, or dip'] != 'Not at all') & (~(df.loc[:,'c. Chewing tobacco, snuff, snus, or dip'].isna())),'c. Chewing tobacco, snuff, snus, or dip'] = 1
    df.loc[(df.loc[:,'c. Chewing tobacco, snuff, snus, or dip'] == 'Not at all'),'c. Chewing tobacco, snuff, snus, or dip'] = 0
    df.loc[:,'tobacco_preg'] = (df.loc[:,'cigarette_pn'] + df.loc[:,'e-cigarette'] 
                                + df.loc[:, 'Hookah'] + df.loc[:, 'd. Cigars, cigarillos, or little filtered cigars'] 
                                + df.loc[:,'c. Chewing tobacco, snuff, snus, or dip'] )

def process_weight_class(df):
    df.loc[(df.loc[:,'weight_class'] == 'Normal weight range (5 pounds 8 ounces to 9 pounds 4 ounces)') |
            (df.loc[:,'weight_class'] =='High birthweight (More than 9 pounds 4 ounces or 4500 grams)'),'weight_class'] = 0
    df.loc[(df.loc[:,'weight_class']== 'Declined to answer (based on response to Question 7)')|
            (df.loc[:,'weight_class']=='Don\'t know (based on response to Question 7)'),'weight_class'] = np.nan
    df.loc[(df.loc[:,'weight_class'] == 'Low birthweight (At least 3 pounds 5 ounces but less than 5 pounds 8 ounces or 2500 grams)'), 'weight_class'] = 1
    df.loc[(df.loc[:,'weight_class'] == 'Very low birthweight (Less than 3 pounds 5 ounces or 1500 grams)'), 'weight_class'] = 2

def recode_twin(twin):
    if twin == 'Singleton (from a pregnancy involving just one baby)':
        return 0
    elif twin == 'Twins':
        return 1
    else:
        return np.nan

def enrol_time(time):
    if time == 'Prior to this pregnancy':
        return 0
    elif time == 'During 1st trimester of this pregnancy (weeks 0-13)':
        return 1
    elif time == 'During 2nd trimester of this pregnancy (14-27)':
        return 2
    elif time == 'During 3rd trimester of this pregnancy (weeks 28-40)':
        return 3
    else:
        return np.nan

def edu_trans(edu):
    if edu == 'Some high school (Grades 9, 10, 11, & 12)' or edu == '8th grade or less':
        return 0
    elif edu == 'High school diploma (Completed 12th grade)' or edu == 'G.E.D.':
        return 1
    elif edu == 'Graduate or professional school' or edu == 'Bachelor\'s degree':
        return 3
    elif edu == edu == 'Some college or 2 year degree' or edu == 'Technical or trade school':
        return 2
    else:
        return np.nan
    
def pre_trans(preventive):
    if preventive == 'No':
        return 0
    elif preventive == 'Don\'t know':
        return np.nan
    else:
        return 1

def inc_trans(income):
    if income == 'Don\'t know' or income == 'Declined to answer':
        return np.nan
    elif isinstance(income, str):
        income = str(income)
        low, high = income.replace(',','').replace('$','').split(' to ')
        return (int(low) + int(high))/2
    else:
        return np.nan

def sub_trans(subs):
    if subs == 'Never':
        return 0
    elif subs == 'Declined to answer':
        return np.nan
    else:
        return 1
    
def prev_trans(cond):
    if cond == 'Yes':
        return 1
    elif cond == 'No':
        return 0
    else:
        return np.nan

def sub_trans(subs):
    if subs == 'Never':
        return 0
    elif subs == 'Declined to answer':
        return np.nan
    else:
        return 1
    
def condom_trans(con):
    if con == 'Yes' or con == 'N/A - not sexually active':
        return 1
    elif con == 'No':
        return 0
    elif con == 'Declined to answer' or con == 'Don\'t know':
        return np.nan

def care_time(time):
    if time == 'First trimester (0-13 weeks)':
        return 0
    elif time == 'Second trimester (14-27 weeks)':
        return 1
    elif time == 'Third trimester (28-40 weeks)':
        return 2
    else:
        return np.nan

def first_preg(info):
    if info == 'This is my first pregnancy':
        return 1
    elif info != 'Don\'t know' or info != 'Declined to answer':
        return 0
    else:
        return np.nan

def spacing_18m(info):
    if info == '0 to 12 months' or info == '13 to 18 months':
        return 0
    elif info != 'Don\'t know' or info != 'Declined to answer':
        return 1
    else:
        return np.nan

def medicaid(ins):
    if isinstance(ins, str):
        choices = ins.split('|')
        for choice in choices:
            if 'Medicaid' in choice or 'CHIP':
                return 1
    return 0

def medicaid(ins):
    if isinstance(ins, str):
        choices = ins.split('|')
        for choice in choices:
            if 'Medicaid' in choice or 'CHIP' in choice:
                return 1
    return 0

def private(ins):
    if isinstance(ins, str):
        choices = ins.split('|')
        for choice in choices:
            if choice == 'Private health insurance from my job or the job of my spouse or partner' or \
                choice == 'Private health insurance from my parents':
                return 1
    return 0

def obamacare(ins):
    if isinstance(ins, str):
        choices = ins.split('|')
        for choice in choices:
            if 'Magnolia' in choice:
                return 1
    return 0

    