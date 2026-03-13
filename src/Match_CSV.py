import pandas as pd

# The main CSV to match with
main_df = pd.read_csv('Data/all_tns_supernovae.csv')

# List of individual CSVs to match with
file_path = 'Data/Individual/'
individual_files = ['emsely.csv', 'gong.csv', 'mingsuwan.csv', 'ruiz.csv']

# Goes through the individual CSVs
for individal in individual_files:
    
    # Turn the individual CSV into a dataframe
    df = pd.read_csv(file_path + individal)
    
    df['sn_id'] = df['sn_id'].astype(str).str.strip()  # Ensure sn_id is a string and remove any leading/trailing whitespace
    main_df['name'] = main_df['name'].astype(str).str.strip()  # Ensure name is a string and remove any leading/trailing whitespace
    
    # Match the individual dataframe with the main dataframe based on the 'Name' column
    out = df.merge(main_df, how='left', left_on='sn_id', right_on='ZTF_ID', suffixes=('', '_main'))

    out = out.loc[:, ~out.columns.str.contains(r"^Unnamed")]
    
    # Convert the dataframe into a csv
    out.to_csv(file_path + 'cleaned_' + individal, index=False)
    
print("All individual CSVs have been matched with the main CSV and saved as new CSVs with the prefix 'cleaned_'.")