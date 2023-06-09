import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu
import seaborn as sns
import matplotlib.patches as  mpatches
import matplotlib.pyplot as plt
np.random.seed(10)

lab_path = 'data\ADNI\ADNI_T1w_axslicez127_list.txt'
path_img = 'data\ADNI\ADNI_T1w_reg2std_axslicez127.nii.gz'
path_matching = 'data/ADNI/matching/all_data/'
path_txt_img = path_matching + 'ADNI_T1w_axslicez127_matched_list.txt'

# lab_path = 'data\ADNI\ADNI_T1w_clean_axslicez127_list.txt'
# path_img = 'data\ADNI\ADNI_T1w_reg2std_axslicez127_clean.nii.gz'
# path_matching = 'data/ADNI/matching/clean_data/'
# path_txt_img = path_matching + 'ADNI_T1w_axslicez127_clean_matched_list.txt'

path_csv = 'data/ADNI/ADNI.csv'
path_removed = path_matching + 'ADNI_removed.csv'
path_matched = path_matching + 'ADNI_matched.csv'
path_count_matched = path_matching + 'ADNI_count_matched.csv'
path_txt = path_matching + 'output_matching.txt'
img_path_before = path_matching + 'Hist_Before_Matching.JPG'
img_path_after = path_matching + 'Hist_After_Matching.JPG'

p_thresholdAD = 0.05
p_thresholdCN = 0.05
itermax = 100

# ----------------------------------------------
min_age_AD = 49
max_age_AD = 60
nM_R_AD = 1
nF_R_AD = 1

min_age_CN = 73
max_age_CN = 83
nM_R_CN = 0
nF_R_CN = 2
# ----------------------------------------------
if not os.path.exists(path_matching):
    os.makedirs(path_matching)

if os.path.exists(path_txt):
    os.remove(path_txt)

f = open(path_txt, "a")
############################################################################################################
#     Create a dataframe with the 'Image Data ID' ordered coherently with the images in the nifti file:
############################################################################################################

# The file at 'lab_path' contains the name of the nifti images from where has been extracted the slices.
# The order of the paths in this file is coherent with the order of the slices saved in 'path_img'.
with open(lab_path) as l:
    labels_txt = l.readlines()

# Create a dataframe that contains the 'Image Data ID' of each image, and the associated path.
lab_df = pd.DataFrame(columns=['ID','path'], index=range(len(labels_txt)))
for i, lab_p in enumerate(labels_txt):
    # Example of lab_p: 'ADNI_002_S_0295_MR_MP-RAGE__br_raw_20060418193713091_1_S13408_I13722_reg2MNI_dof9_conform_reoriented_axslicez127.nii.gz\n'
    p = Path(lab_p).stem
    parts = p.split('_') # ['ADNI', '002', 'S', '0295', 'MR', 'MP-RAGE', '', 'br', 'raw', '20060418193713091', '1', 'S13408', 'I13722', 'reg2MNI', ...]
    lab_df['ID'][i] = parts[-6] # 'I13722'
    lab_df['path'][i] = lab_p

# Load the dataframe with some metadata:
ADNI_df = pd.read_csv(path_csv)

# Merge 'lab_df' and 'ADNI_df' in order to re-order the lines with the same order of the slices saved in 'path_img'
merged_df = lab_df.merge(ADNI_df, left_on=lab_df['ID'], right_on=ADNI_df['Image Data ID'], how='left')
# Note: To verify that the condition 'merged_df['path'] == labels_txt' is verified for every line: 
#       sum(merged_df['path'] == labels_txt) == len(labels_txt)

############################################################################################################
#                                      Check age and sex matching:
############################################################################################################
AD_df = merged_df[merged_df['Group']=='AD']
CN_df = merged_df[merged_df['Group']=='CN']

# Visualize the distribution of the age to decide in which range remove controls:
age_AD = AD_df['Age']
age_CN = CN_df['Age']


print('\nLegend:', file=f)
print('T: p-value of T-test - to test the AGE matching', file=f)
print('MW: p-value of Mann-Withney U test - to test the AGE matching', file=f)
print('chi: p-value of the chi-square test - to test the SEX matching', file=f)
print('---------------------------------------------------------------', file=f)

####################################################################################################################
#                                                  REMOVE AD:
####################################################################################################################

plt.close('all')
# sns.kdeplot(age_AD, shade=True, legend=True, color="red", cmap="Reds", common_norm=False)
# sns.kdeplot(age_CN, shade=True, legend=True, cmap="Blues", common_norm=False)
sns.histplot(data=age_AD, binwidth=1, kde=True, legend=True, color="red")
sns.histplot(data=age_CN, binwidth=1, kde=True, legend=True, color="blue")
handles = [mpatches.Patch(facecolor=plt.cm.Reds(100), label='AD'),
        mpatches.Patch(facecolor=plt.cm.Blues(100), label='CN')]
plt.title('Age distribution before matching')
# plt.title('Decide the range from where remove AD')
plt.legend(loc='best', handles=handles)
plt.xlim(0,100)
plt.ylim(0,max(CN_df.groupby(['Age']).count()['ID'])+5)
plt.savefig(img_path_before)
#plt.show()

print('\nSet the range of age in which remove AD patients:', file=f)
# min_age = input('Min: ')
# max_age = input('Max: ')
print('[{}, {}]'.format(min_age_AD,max_age_AD), file=f)

nM_AD = sum(AD_df['Sex']=='M')
nM_CN = sum(CN_df['Sex']=='M')
nF_AD = sum(AD_df['Sex']=='F')
nF_CN = sum(CN_df['Sex']=='F')

print('\nNumber of AD males: {}\nNumber of AD females: {}'.format(nM_AD, nF_AD), file=f)
print('Number of CN males: {}\nNumber of CN females: {}'.format(nM_CN, nF_CN), file=f)
# nM_R = input('Number of AD males to remove at each iteration: ')
# nF_R = input('Number of AD females to remove at each iteration: ')
print('\nIn each iteration remove {} M and {} F from AD.'.format(nM_R_AD, nF_R_AD), file=f)

tStat, T_pValue = ttest_ind(age_AD, age_CN, equal_var = False)
MW = mannwhitneyu(age_AD, age_CN)  
MW_pValue = MW.pvalue    
print('\nOriginal: T = {}'.format(T_pValue), file=f)
print('         MW = {}'.format(MW_pValue), file=f)
contigency_train = pd.crosstab(merged_df['Sex'], merged_df['Group'])
c, pvalue_sex_matched, dof, expected = chi2_contingency(contigency_train)
print('        chi2 = {}\n'.format(pvalue_sex_matched), file=f)


new_df = merged_df.copy()
rm_df = pd.DataFrame(columns=merged_df.columns)

# REMOVE AD PATIENTS:
i=0
while (min(T_pValue,MW_pValue)<p_thresholdAD) & (i<itermax):

    # Separate males and females:
    df_M = AD_df.query("Sex=='M'")
    df_F = AD_df.query("Sex=='F'")

    # Extract patients with AGE in the range [min_age, max_age]: 
    df_M_selected = df_M.query("{} < Age < {}".format(min_age_AD, max_age_AD))
    df_F_selected = df_F.query("{} < Age < {}".format(min_age_AD, max_age_AD))
    
    # Select the patients to be removed and memorize them
    df_M_removed = df_M_selected.sample(n=int(nM_R_AD))
    df_F_removed = df_F_selected.sample(n=int(nF_R_AD))
    rm_df = pd.concat([rm_df,df_M_removed]) 
    rm_df = pd.concat([rm_df,df_F_removed])

    # Remove the selected subject from the CONT of the GROUP of interest:
    rm_id = pd.DataFrame(rm_df['ID']).rename(columns={"ID": "ID_rm"})
    new_df = merged_df.merge(rm_id, left_on='ID', right_on='ID_rm', how='left', indicator=True)
    new_df = new_df[new_df['_merge'] == 'left_only']

    # Update p-value for the new dataframe:
    age_AD = new_df[new_df['Group']=='AD']['Age']
    tStat, T_pValue = ttest_ind(age_AD, age_CN, equal_var = True)
    MW = mannwhitneyu(age_AD, age_CN)  
    MW_pValue = MW.pvalue   
    print('Iteration ', i, ': T = {}'.format(T_pValue), file=f)
    print('               MW = {}'.format(MW_pValue), file=f)
    contigency_train = pd.crosstab(new_df['Sex'], new_df['Group'])
    c, pvalue_sex_matched, dof, expected = chi2_contingency(contigency_train)
    print('             chi2 = {}'.format(pvalue_sex_matched), file=f)
    i+=1


####################################################################################################################
#                                                  REMOVE CN:
####################################################################################################################
# sns.kdeplot(age_AD, shade=True, legend=True, color="red", cmap="Reds")
# sns.kdeplot(age_CN, shade=True, legend=True, cmap="Blues")
sns.histplot(data=age_AD, binwidth=1, kde=True, legend=True, color="red")
sns.histplot(data=age_CN, binwidth=1, kde=True, legend=True, color="blue")
handles = [mpatches.Patch(facecolor=plt.cm.Reds(100), label='AD'),
        mpatches.Patch(facecolor=plt.cm.Blues(100), label='CN')]
plt.title('Decide the range in which remove CN subjects:')
plt.legend(loc='best', handles=handles)
plt.xlim(0,100)
plt.ylim(0,max(CN_df.groupby(['Age']).count()['ID'])+5)
#plt.show()

print('\nSet the range of age in which remove CN subjects:', file=f)
# min_age = input('Min: ')
# max_age = input('Max: ')
print('[{}, {}]'.format(min_age_CN, max_age_CN), file=f)

nM_AD = sum(AD_df['Sex']=='M')
nM_CN = sum(CN_df['Sex']=='M')
nF_AD = sum(AD_df['Sex']=='F')
nF_CN = sum(CN_df['Sex']=='F')

print('\nNumber of AD males: {}\nNumber of AD females: {}'.format(nM_AD, nF_AD), file=f)
print('Number of CN males: {}\nNumber of CN females: {}'.format(nM_CN, nF_CN), file=f)
# nM_R = input('Number of CN males to remove at each iteration: ')
# nF_R = input('Number of CN females to remove at each iteration: ')
print('In each iteration remove {} M and {} F from CN.\n'.format(nM_R_CN, nF_R_CN), file=f)

i=0
while (min(T_pValue,MW_pValue, pvalue_sex_matched)<p_thresholdCN) & (i<itermax):
    # Separate males and females:
    df_M = CN_df.query("Sex=='M'")
    df_F = CN_df.query("Sex=='F'")

    # Extract patients with AGE in the range [min_age, max_age]: 
    df_M_selected = df_M.query("{} < Age < {}".format(min_age_CN, max_age_CN))
    df_F_selected = df_F.query("{} < Age < {}".format(min_age_CN, max_age_CN))
    
    # Select the patients to be removed and memorize them
    df_M_removed = df_M_selected.sample(n=int(nM_R_CN))
    df_F_removed = df_F_selected.sample(n=int(nF_R_CN))
    rm_df = pd.concat([rm_df,df_M_removed]) 
    rm_df = pd.concat([rm_df,df_F_removed])

    # Remove the selected subject from the CONT of the GROUP of interest:
    rm_id = pd.DataFrame(rm_df['ID']).rename(columns={"ID": "ID_rm"})
    new_df = merged_df.merge(rm_id, left_on='ID', right_on='ID_rm', how='left', indicator=True)
    new_df = new_df[new_df['_merge'] == 'left_only']

    # Update p-value for the new dataframe:
    age_CN = new_df[new_df['Group']=='CN']['Age']
    tStat, T_pValue = ttest_ind(age_AD, age_CN, equal_var = True)
    MW = mannwhitneyu(age_AD, age_CN)  
    MW_pValue = MW.pvalue   
    print('Iteration ', i, ': T = {}'.format(T_pValue), file=f)
    print('               MW = {}'.format(MW_pValue), file=f)
    contigency_train = pd.crosstab(new_df['Sex'], new_df['Group'])
    c, pvalue_sex_matched, dof, expected = chi2_contingency(contigency_train)
    print('             chi2 = {}'.format(pvalue_sex_matched), file=f)
    i+=1

plt.close('all')
# sns.kdeplot(age_AD, shade=True, legend=True, color="red", cmap="Reds")
# sns.kdeplot(age_CN, shade=True, legend=True, cmap="Blues")
sns.histplot(data=age_AD, binwidth=1, kde=True, legend=True, color="red")
sns.histplot(data=age_CN, binwidth=1, kde=True, legend=True, color="blue")
handles = [mpatches.Patch(facecolor=plt.cm.Reds(100), label='AD'),
        mpatches.Patch(facecolor=plt.cm.Blues(100), label='CN')]
plt.title('Age distribution after matching')
plt.legend(loc='best', handles=handles)
plt.xlim(0,100)
plt.ylim(0,max(CN_df.groupby(['Age']).count()['ID'])+5)
#plt.show()
plt.savefig(img_path_after)

# Save themathed data in csv:
rm_df.to_csv(path_removed)
new_df.to_csv(path_matched)

# SUBJECTS COUNT:

images_xSubject = new_df.groupby(by=['Subject']).count()['Image Data ID']
images_xSubject.to_csv(path_count_matched)
n_subjects = len(images_xSubject)
n_AD = len(new_df[new_df['Group']=='AD'].groupby(by=['Subject']).count())
n_AD_images = len(new_df[new_df['Group']=='AD'])
n_CN = len(new_df[new_df['Group']=='CN'].groupby(by=['Subject']).count())
n_CN_images = len(new_df[new_df['Group']=='CN'])
n_AD_rm_img = len(rm_df[rm_df['Group']=='AD'])
n_AD_rm = len(rm_df[rm_df['Group']=='AD'].groupby(by=['Subject']).count())
n_CN_rm_img = len(rm_df[rm_df['Group']=='CN'])
n_CN_rm = len(rm_df[rm_df['Group']=='CN'].groupby(by=['Subject']).count())

print('\n', file=f)
print('Total number of subjects before matching: {}'.format(len(merged_df.groupby(by=['Subject']).count())), file=f)
print('Total number of subjects after matching: {} di cui AD patients: {} e CN subjects: {}'.format(len(new_df.groupby(by=['Subject']).count()), n_AD,n_CN), file=f)
print('\n', file=f)
print('Total number of images before matching: {}'.format(len(merged_df)), file=f)
print('Total number of images after matching: {} di cui AD images: {} e CN images: {}'.format(len(new_df), n_AD_images, n_CN_images), file=f)
print('The {} images ({} AD, {} CN) removed for age- and sex-matching between CN subjects and AD patients belong to {} subjects ({} AD patients and {} CN subjects)'.format(len(rm_df), n_AD_rm_img, n_CN_rm_img, len(rm_df.groupby(by=['Subject']).count()), n_AD_rm, n_CN_rm), file=f)

f.close()

# -----------------------------------
# Save the paths pf the image to concatenate:
if os.path.exists(path_txt_img):
    os.remove(path_txt_img)

img_paths = open(path_txt_img, "a")
for i in new_df['path'].index:
    print(new_df['path'][i][:-1], file=img_paths)
img_paths.close()