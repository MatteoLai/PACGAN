"""
Non-Profit Open Software License 3.0 (NPOSL-3.0)
Copyright (c) 2023 Matteo Lai

@authors: 
         Matteo Lai, M.Sc, Ph.D. student
         Dept. of Electrical, Electronic and Information Engineering – DEI "Guglielmo Marconi",
         University of Bologna, Bologna, Italy.
         Email address: matteo.lai3@unibo.it

         Chiara Marzi, Ph.D., Junior Assistant Professor
         Department of Statistics, Computer Science, Applications "G. Parenti" (DiSIA)
         University of Firenze, Firenze, Italy
         Email address: chiara.marzi@unifi.it
         
         Luca Citi, Ph.D., Professor
         School of Computer Science and Electronic Engineering
         University of Essex, Colchester, UK
         Email address: lciti@essex.ac.uk
         
         Stefano Diciotti, Ph.D, Associate Professor
         Dept. of Electrical, Electronic and Information Engineering – DEI "Guglielmo Marconi",
         University of Bologna, Bologna, Italy.
         Email address: stefano.diciotti@unibo.it
"""

import pandas as pd
import nibabel as nib
import numpy as np
import os
import skimage.transform as skTrans
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu
import seaborn as sns
import matplotlib.patches as  mpatches
import matplotlib.pyplot as plt
np.random.seed(1)

##########################################################################
# Set the sizes of the images you will use to train the PACGAN:
sizes = [4, 8, 16, 32, 64, 128, 256]
##########################################################################

path_list = 'data\ADNI\ADNI_T1w_axslicez127_clean_matched_list.txt'
path_img = 'data\ADNI\ADNI_T1w_reg2std_axslicez127_clean_matched.nii.gz'
path_csv = 'data/ADNI/ADNI.csv'

def load_nifti(path):
    img = nib.load(path)
    img = np.asanyarray(img.dataobj) # numpy array of shape (W,H,C,N)
    img = np.float32(img)
    return img

def extract_slice(vol, vol_t, i, cnt):
    img_slice = vol[:,:,:,i]
    mx = img_slice.max()
    vol_t[:,:,:,cnt]  = (img_slice - mx/2)/(mx/2) # Rescale in [-1,1]
    cnt+=1
    return cnt

def create_folder(path):
    '''If the folder doesn't exist, create it.'''
    if not os.path.isdir(path): 
        os.makedirs(path)
        print('Created folder: ', path)

def save_nifti(img, images_path, image_name):
    nifti_img = nib.Nifti1Image(img, np.eye(4))
    create_folder(images_path)
    nib.save(nifti_img, os.path.join(images_path, image_name))

def resize_images(images_path, image_name, sizes):
    img = load_nifti(images_path+image_name)
    for size in sizes[:-1]:
        img_resized = skTrans.resize(img, (size, size, img.shape[2], img.shape[3]), order=1, preserve_range=True)
        save_nifti(img_resized, images_path, "Real{}x{}.nii.gz".format(size,size))

############################################################################################################
#     Create a dataframe with the 'Image Data ID' ordered coherently with the images in the nifti file:
############################################################################################################

# The file at 'path_list' contains the name of the nifti images from where has been extracted the slices.
# The order of the paths in this file is coherent with the order of the slices saved in 'path_img'.
with open(path_list) as l:
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

####################################################################################################
#       Divide the subjects in training and test set paying attention to the class balance:
####################################################################################################
# ------------------------------------------------------------------------------------------------
# Hold out:
# To keep 20% of data for the test set, perform the 5-fold CV
X = merged_df['Image Data ID']
y = merged_df['Group']
groups = merged_df['Subject']
cv = StratifiedGroupKFold(n_splits=5)
for i in range(100):
    print(i)
    for train_idxs, test_idxs in cv.split(X, y, groups):
        train_img = X[train_idxs]
        subjects_train = groups[train_idxs]
        train_y = y[train_idxs]
        test_img = X[test_idxs]
        subjects_test = groups[test_idxs]
        test_y = y[test_idxs]

        # Define one dataframe for each group:
        train_df = pd.DataFrame(subjects_train).rename({0:'Subject'}, axis='columns')
        train_df = train_df.groupby(by=['Subject']).sum()
        train_df = train_df.merge(merged_df, how='left', left_on='Subject', right_on='Subject')

        test_df = pd.DataFrame(subjects_test).rename({0:'Subject'}, axis='columns')
        test_df = test_df.groupby(by=['Subject']).sum()
        test_df = test_df.merge(merged_df, how='left', left_on='Subject', right_on='Subject')

        # Age matching:
        # -> train:
        age_AD_train = train_df[train_df['Group']=='AD']['Age']
        age_CN_train = train_df[train_df['Group']=='CN']['Age']
        tStat, Tpvalue_age_train = ttest_ind(age_AD_train, age_CN_train, equal_var = False)
        MW = mannwhitneyu(age_AD_train, age_CN_train)  
        MWpvalue_age_train = MW.pvalue 
        # -> test:
        age_AD_test = test_df[test_df['Group']=='AD']['Age']
        age_CN_test = test_df[test_df['Group']=='CN']['Age']
        tStat, Tpvalue_age_test = ttest_ind(age_AD_test, age_CN_test, equal_var = False)
        MW = mannwhitneyu(age_AD_test, age_CN_test)  
        MWpvalue_age_test = MW.pvalue   
        # Sex matching:
        # -> train:
        contigency_train = pd.crosstab(train_df['Sex'], train_df['Group'])
        c, pvalue_sex_train, dof, expected = chi2_contingency(contigency_train)
        # -> test:
        contigency_test = pd.crosstab(test_df['Sex'], test_df['Group'])
        c, pvalue_sex_test, dof, expected = chi2_contingency(contigency_test) 

        th = 0.05
        # condition_age = Tpvalue_age_train>th and Tpvalue_age_test>th and MWpvalue_age_train>th and MWpvalue_age_test>th
        condition_age = MWpvalue_age_train>th and MWpvalue_age_test>th
        condition_sex = pvalue_sex_train>th and  pvalue_sex_test>th
        if condition_age and condition_sex:
            break
    if condition_age and condition_sex:
        break

# ------------------------------------------------------------------------------------------------

# # To be sure that the same subject do not appear in both train and test set:
# for sub1 in train_sub:
#     for sub2 in test_sub:
#         if sub1==sub2:
#             print(sub1)

# ------------------------------------------------------------------------------------------------

# print('Age matching: \nTrain T: {}\nTrain MW: {}\nTest T: {}\nTest MW: {}'.format(Tpvalue_age_train, MWpvalue_age_train, Tpvalue_age_test, MWpvalue_age_test))
print('Age matching: \nTrain MW: {}\nTest MW: {}'.format(MWpvalue_age_train, MWpvalue_age_test))
print('Sex matching: \nTrain: {}\nTest: {}'.format(pvalue_sex_train, pvalue_sex_test))

plt.subplot(121)
# Plot the histograms of age distributions for training e test set:
sns.histplot(data=age_AD_train, binwidth=1, kde=True, legend=True, color="red")
sns.histplot(data=age_CN_train, binwidth=1, kde=True, legend=True, color="blue")
handles = [mpatches.Patch(facecolor=plt.cm.Reds(100), label='AD'),
        mpatches.Patch(facecolor=plt.cm.Blues(100), label='CN')]
plt.title('Train')
# plt.title('Decide the range from where remove AD')
plt.legend(loc='best', handles=handles)
plt.xlim(0,100) 

#
plt.subplot(122)
sns.histplot(data=age_AD_test, binwidth=1, kde=True, legend=True, color="red")
sns.histplot(data=age_CN_test, binwidth=1, kde=True, legend=True, color="blue")
handles = [mpatches.Patch(facecolor=plt.cm.Reds(100), label='AD'),
        mpatches.Patch(facecolor=plt.cm.Blues(100), label='CN')]
plt.title('Test')
# plt.title('Decide the range from where remove AD')
plt.legend(loc='best', handles=handles)
plt.xlim(0,100) 

plt.show()


####################################################################################################
#                  Separate the training and test images in two nifti files:
####################################################################################################

vol = load_nifti(path_img)
vol_train = np.zeros((sizes[-1], sizes[-1], 1, len(train_df)))
vol_test = np.zeros((sizes[-1], sizes[-1], 1, len(test_df)))

cnt_train=0; cnt_test=0

for i, ID in enumerate(merged_df['ID'].tolist()):
    # Train:
    if ID in train_df['ID'].tolist():
        cnt_train = extract_slice(vol, vol_train, i, cnt_train)

    #Test:
    elif ID in test_df['ID'].tolist():
        cnt_test = extract_slice(vol, vol_test, i, cnt_test)

    else:
        print('Error: '+ ID)

# Save nifti files:
p1 = Path(path_img)
nifti_path = p1.parents[0]
image_name = 'Real256x256.nii.gz'

# -> Train:
train_folder = str(nifti_path)+'/ADNI_train/'; create_folder(train_folder)
save_nifti(vol_train, train_folder, image_name)
resize_images(train_folder, image_name, sizes)

# -> Test:
test_folder = str(nifti_path)+'/ADNI_test/'; create_folder(test_folder)
save_nifti(vol_test, test_folder, image_name)
resize_images(test_folder, image_name, sizes)

####################################################################################################
#             Define the csv with the labels for the training and test images:
####################################################################################################

# Train:
labels_train = pd.DataFrame(columns=['ID','Label', 'Subject_ID'], index=range(len(train_df)))
labels_train['ID'] = train_df.index
labels_train['Label'] = train_df.Group.replace({"AD":1, "CN":0})
labels_train['Subject_ID'] = train_df['Subject']
labels_train.to_csv(train_folder+'/labels.csv', index=False)

# Test:
labels_test = pd.DataFrame(columns=['ID','Label', 'Subject_ID'], index=range(len(test_df)))
labels_test['ID'] = test_df.index
labels_test['Label'] = test_df.Group.replace({"AD":1, "CN":0})
labels_test['Subject_ID'] = test_df['Subject']
labels_test.to_csv(test_folder+'/labels.csv', index=False)
