import pandas as pd
import numpy as np
from .data_processing import classifer, housing, trimester

def load_bg(filename):
    '''
    loading background form
    '''
    df = pd.read_csv(filename)
    
    bg = df.rename(columns = {'Name of Primary Participant:':'Name_PP',
                           'Primary Participant Date of Birth:':'DOB_PP',
                           'WHAT PHASE IS THIS FORM CURRENTLY BEING COMPLETED FOR (HRSA: USED FOR G7-G9)': 'form_purpose',
                           'G4. THIS PARTICIPANT HAS AT LEAST ONE ENROLLED CHILD ATTACHED TO HER/HIM:':'enrolled_child',
                    'G6. WHAT PHASE OF THE REPRODUCTIVE CYCLE WAS THE PRIMARY PARTICIPANT IN WHEN HE/SHE FIRST ENROLLED IN HS?':'repro_cycle',
                    '1. What is your sex? Select one.':'Sex',
                    '2. Now I\'d like to ask some questions about your education. What is the highest grade or level of school that you have completed?':'highest_edu',
                    '5. Which ONE racial classification below do you identify with the most? [Select one.]':'race',
                   '8. During the past 12 MONTHS, were you EVER covered by ANY kind of health insurance or health coverage plan?':'insurance coverage',
                           '10. During the past 12 months, did you see a doctor, nurse, or other health care professional for PREVENTIVE medical care, such as a physical or well-visit checkup? A preventive check-up is when you are not sick or injured, such as an annual or sports physical, or well-visit. [Select one.]  [Staff: A visit for PREVENTIVE medical care DOES NOT include prenatal care]':'preventive',
                           
                    '11. First, can you tell me, during the past 12 months, what was your yearly total household income before taxes? Please include all sources of income, including your income, your spouse’s or partner’s income, and any other income you may have received. All information will be kept private and will not affect any services you are now getting. [Select one.]':'household income',
                     '12. During the past 12 months, how many people, including yourself, depended on this income? [Staff: Enter number of people.]':'household size',
                    'TOTAL SCORE':'mental health',
                    'Used any tobacco product (for example, cigarettes, ecigarettes, cigars, pipes, or smokeless tobacco)?':'tobacco',
                    'For women: Had 4 or more drinks containing alcohol in one day? For men: Had 5 or more drinks containing alcohol in one day? One standard drink is about 1 small glass of wine (5 oz), 1 beer (12 oz), or 1 single shot of liquor.':'alcohol',
                    'Used marijuana?':'marijuana',
                    'Used any illicit drugs including cocaine or crack, heroin, methamphetamine (crystal meth), hallucinogens, ecstasy/MDMA?':'illicit drugs',
                           '25. All participants... Are you currently using a condom to prevent sexually transmitted infections? [Select one.]':'condom',
                    '27. Are you pregnant now? [Select one.]':'Pregnancy',
                    'Pregnancy that did not result in a live birth (disabled)_27088':'prev_non_live_birth',
                           'Miscarriage, number':'prev_miscarriage',
                    '29. A preterm delivery is one that occurs before the 37th week of pregnancy. As far as you know, have you had any preterm delivery in the past? [Select one.]':'prev_preterm',
                     '30. Did any of your babies weigh LESS than 5 pounds, 8 ounces [2500 grams] at birth? Select one.':'prev_LBW',
                    '31. [Staff: skip this question if mother has not had previous babies born less than 5 lb, 8 oz] Thinking about your babies who were born weighing less than 5 pounds, 8 ounces, how many of them weighed less than 3 pounds, 5 ounces [1500 grams] at birth? Select one.':'prev_very_LBW',
                    '33. Did any of your babies stay in the hospital after you came home? [Select one.]':'prev_hospitalized_birth',
                          '9. What kind of health insurance do you have now? [Select all that apply.]':'insurance'
                    })
    
    #selecting features
    bg1 = bg.loc[bg.loc[:,'Program'] == 'DHSC Delta Healthy Start Collaborative',['Subject Name', 'SubjectID', 'Name_PP','DOB_PP','form_purpose',
                                                                              'Response Date','enrolled_child', 'household size',
                'repro_cycle', 'Sex','highest_edu','race','preventive','household income', 'mental health',
                'tobacco', 'alcohol','marijuana','illicit drugs','condom','Pregnancy','prev_miscarriage',
                'prev_preterm','prev_LBW','prev_very_LBW','prev_hospitalized_birth','insurance']]
    
    return bg1

def load_pc(filename):
    '''
    loading parent child form
    '''
    pc = pd.read_csv('./ETO_parent_child.csv')
    
    #change variable name
    var_name = {'Name of Primary Participant':'Name_PP',
                'Primary Participant DOB:':'DOB_PP',
         'G2. Enrolled Child\'s DOB:':'child_bday',
         'G9. Child is:' : 'child gender',
         '4. Which ONE racial classification below do you think best describes your child\'s racial background? Select ONE only.':'race',
         '5. How many weeks pregnant were you [was the mother] when he/she was born? [STAFF: Please enter number of weeks. If participant does not know number of weeks, help them calculate backwards from the baby’s original due date to determine weeks gestation at birth.]' : 'preg_weeks',
         'Calculated birth weight in grams:':'birth_weight_gram',
         '8. [Staff: Please check appropriate box below for baby\'s birthweight]:':'weight_class',
         '9. Was this child the only baby you were [the mother was] pregnant with at the time, or was it a multiple birth, such as twins, triplets, more?':'singleton',
         '12. DURING THE PAST 12 MONTHS, was the child EVER covered by ANY kind of health insurance or health coverage plan?': 'child_coverage',
        '24. In the last 3 months of your pregnancy with this child, how many cigarettes did you smoke on an average day? A pack has 20 cigarettes.':'cigarette'
    }
    pc1 = pc.rename(columns = var_name)
    
    #picking features
    pc = pc1.loc[pc1.loc[:,'Program'] == 'DHSC Delta Healthy Start Collaborative',[
        'Subject Name', 'SubjectID','Response Date', 'Response ID',
        'Name_PP','DOB_PP','child_bday','child gender',
        'race','preg_weeks','birth_weight_gram',
        'weight_class','singleton','child_coverage',
        y'cigarette']]
    return pc

def load_pn(filename):
    '''
    loading prenatal form
    '''
    pn = pd.read_csv(filename)
    
    var_name = {'Name of Primary Participant:':'Name_PP',
            'Primary Participant Date of Birth':'DOB_PP', 
            '3. [Staff, based on how many weeks pregnant the woman is, what trimester is she currently in?]' :'trimester',
            '4. [Staff, When did the participant enroll?]':'enrollment_time',
            '6. [Staff: Please select corresponding trimester for when woman had her first prenatal care visit]:':'first_prenatal_care_time',
             'a. Type 1 or Type 2 diabetes (not gestational diabetes or diabetes that starts during pregnancy)':'diabetes',
             'b. High blood pressure or hypertension':'hypertension',
            'c. Depression':'depression',
            '8. [Staff, if mother currently has another child besides the one she is pregnant with, ask:] Thinking about your child who was born just before the one you\'re pregnant with, how old was he/she when you learned about this pregnancy?':'prev_kid_age',
            '11. How many cigarettes are you smoking now on an average day? A pack has 20 cigarettes':'cigarette',
            'a. E-cigarettes or other electronic nicotine products':'e-cigarette',
            '13. Since you found out you were pregnant, how often have you been drinking alcoholic beverages?':'alcohol_preg'
           }
    df1 = pn.rename(columns = var_name)
    
    pn1 = df1.loc[df1.loc[:,'Program'] == 'DHSC Delta Healthy Start Collaborative',
                  ['Subject Name', 'SubjectID', 'Program', 'Response Date','Response ID', 
                    'Name_PP', 'DOB_PP','trimester','enrollment_time',
                    'first_prenatal_care_time','diabetes','hypertension',
                    'depression','prev_kid_age','cigarette','e-cigarette',
                    'alcohol_preg','b. Hookah','c. Chewing tobacco, snuff, snus, or dip',
                    'd. Cigars, cigarillos, or little filtered cigars']]
    return pn1

def clean_multi_record(df):
    '''
    removing repeated record for each child
    '''
    #count number of records for each parent and child birthday
    ct = df.groupby(['SubjectID','child_bday']).agg({'child_bday':['count']}).reset_index()
    
    m1 = pd.merge(df, ct, how = 'left', left_on = ['SubjectID','child_bday'], right_on = [( 'SubjectID',''),('child_bday','')])
    
    #eliminating repeated record for the same child - based on parent, child birthday, and birthweight
    m1.loc[:,'ind'] = 1
    m1.rename(columns = {('child_bday', 'count'):'count'}, inplace = True)
    m1.loc[(m1.loc[:,'count'] > 1) & (m1.singleton == 'Singleton (from a pregnancy involving just one baby)'), 'ind'] = 2
    m1.loc[(m1.loc[:,'count'] > 2) & (m1.singleton == 'Twins'), 'ind'] = 3
    m1.loc[:,'entries'] = m1.count(axis = 1)
    m2 = m1.loc[m1.loc[:,'ind'] == 1,:]
    #twins with multiple records
    m3 = m1.loc[(m1.loc[:,'ind'] == 3) ,:].sort_values('entries', ascending = False
                                                   ).drop_duplicates(subset = ['SubjectID','child_bday','birth_weight_gram'], keep = 'first')
    #singleton with multiple records
    m4 = m1.loc[(m1.loc[:,'ind'] == 2) ,:].sort_values('entries', ascending = False
                                                   ).drop_duplicates(subset = ['SubjectID','child_bday'], keep = 'first')
    pc1 = pd.concat([m2,m3,m4])
    
    return pc1

def load_hv(filename):
    '''
    loading home visit data
    '''
    hv= pd.read_csv(filename)
    
    #counting number of response id and values in each row
    hv.groupby('Response ID').agg({'Response ID':['count']}).sort_values(('Response ID','count'))
    hv.loc[:,'entries'] = hv.count(axis = 1)
    
    hv1 = hv.sort_values(['Response ID','entries'], ascending = False
                        ).drop_duplicates(subset = ['Response ID'], keep = 'first')
    
    hv1.drop_duplicates(inplace = True)
    var_name = {'Number of weeks pregnant at time of visit':'preg_week_visit',
            'A. Is the participant on track with weight gain? (not too much, not too little for their stage of pregnancy)':'weight_gain_on_track',
            'B. Has a health professional informed the participant that they have gestational diabetes (diabetes that occurs during pregnancy)?':'gestational diabetes',
            'C. Has a health professional told the participant that they have preeclampsia (high blood pressure or protein in the urine)?':'preeclampsia',
            'D. Has the participant used any tobacco products while pregnant?':'tobacco_con',
            'E. Has the participant drank alcohol (including wine, beer, liquor, homemade liquor) while pregnant?':'alcohol_con',
            'F. Has the participant ingested, used, or injected any products unsafe for pregnancy (drugs, medicines, etc.)?':'unsafe_product',
            'G. Does the person feel comfortable and safe with everyone who is staying in the home/apartment building with them?':'feel_safe',
            'H. Has the participant been diagnosed with any infectious disease (including sexually transmitted diseases) that they need/needed medication while pregnant?':'infection',
            'I. Has any healthcare professional shared any concern about cervix or uterine issues for the participant?':'uterine_issue',
            'J. Over the past week, did you get at least 6 hours of sleep each day?':'sleep_suf',
            'K. Does the participant need any social service assistance, such as food stamps, WIC, TANF, SNAP Benefits, assistance with utility payments, rent?':'social_assistance',
            'L. Has there been a loss, decrease, or gain in income or employment in the household (for example, job loss or spouse/partner losing job)?':'inc_change',
            'M. Is the participant having trouble regulating emotions?':'emo_reg',
            'V. Does the participant currently have employment or a source of income?':'employed',
            'W. Has the Participant experienced a death among their family or loved ones over the past 30 days?':'loss_love',
            'X. Does the participant have a form of transportation that allows them to get to where they want when they want? (for example, do they usually have gas and a car available when they want?)':'transportation',
            'Y. Do you own a place, rent a place, live in public housing, stay with a family, or somwhere else?':'housing',
            'A1. Does the Participant have a library card?':'library_card',
            
           }
    hv1= hv1.rename(columns = var_name)
    
    #selecting only healthy start participants who are pregnant
    hv2 = hv1.loc[(hv1.loc[:,'Is mother pregnant?'] == 'Yes') & (hv1.loc[:, 'Program'] == 'DHSC Delta Healthy Start Collaborative'),[
    'Response ID','Subject Name','SubjectID','Response Date','preg_week_visit','weight_gain_on_track','gestational diabetes',
    'preeclampsia','tobacco_con','alcohol_con','unsafe_product','feel_safe','infection','uterine_issue','sleep_suf',
    'social_assistance','inc_change','emo_reg','employed','transportation','housing','library_card']]
    
    hv2.loc[:,'weight_gain_on_track'] = hv2.loc[:,'weight_gain_on_track'].apply(classifer)
    hv2.loc[:,'gestational diabetes'] = hv2.loc[:,'gestational diabetes'].apply(classifer)
    hv2.loc[:,'preeclampsia'] = hv2.loc[:,'preeclampsia'].apply(classifer)
    hv2.loc[:,'tobacco_con'] = hv2.loc[:,'tobacco_con'].apply(classifer)
    hv2.loc[:,'alcohol_con'] = hv2.loc[:,'alcohol_con'].apply(classifer)
    hv2.loc[:,'unsafe_product'] = hv2.loc[:,'unsafe_product'].apply(classifer)
    hv2.loc[:,'feel_safe'] = hv2.loc[:,'feel_safe'].apply(classifer)
    hv2.loc[:,'infection'] = hv2.loc[:,'infection'].apply(classifer)
    hv2.loc[:,'uterine_issue'] = hv2.loc[:,'uterine_issue'].apply(classifer)
    hv2.loc[:,'sleep_suf'] = hv2.loc[:,'sleep_suf'].apply(classifer)
    hv2.loc[:,'social_assistance'] = hv2.loc[:,'social_assistance'].apply(classifer)
    hv2.loc[:,'emo_reg'] = hv2.loc[:,'emo_reg'].apply(classifer)
    hv2.loc[:,'employed'] = hv2.loc[:,'employed'].apply(classifer)
    hv2.loc[:,'transportation'] = hv2.loc[:,'transportation'].apply(classifer)
    hv2.loc[:,'housing'] = hv2.loc[:,'housing'].apply(housing)
    hv2.loc[:,'library_card'] = hv2.loc[:,'library_card'].apply(classifer)
    hv2.loc[:,'trimester_visit'] = hv2.loc[:,'preg_week_visit'].apply(trimester)
    
    hv2.to_csv('./cleaned_data/cleaned_hv.csv')

