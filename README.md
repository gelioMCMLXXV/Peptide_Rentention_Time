# Peptide_Rentention_Time
The peptide_retention_time_V0.py is a baby version of a code to learn elution time of peptides in LC-MS/MS experiment.
The inputs for the code are two files one used for taining the other for test. Both files are located in the folder Test_Data.
For the code to run properly the path for the location of the files need to be set corretly in the peptide_retention_time_V0.py code.  
Also there are 2 options in the code that need to be manually changed:
###Code Parameters: option, evalue_cutoff
option=0  ###option 0==elemental composition, 1==Peptide Hydrophobicity Eisenberg
evalue_cutoff = 0.001   ###evalue_cutoff used to filter the peptides identifed 

This code runs without problems in Google Colab Notebook.
