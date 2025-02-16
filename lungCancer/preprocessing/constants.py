# Define column types
CATEGORICAL_COLUMNS = [
    'Country', 'Gender', 'Smoking_Status', 'Second_Hand_Smoke',
    'Occupation_Exposure', 'Rural_or_Urban', 'Socioeconomic_Status',
    'Healthcare_Access', 'Insurance_Coverage', 'Screening_Availability',
    'Stage_at_Diagnosis', 'Cancer_Type', 'Mutation_Type', 'Treatment_Access',
    'Clinical_Trial_Access', 'Language_Barrier', 'Family_History',
    'Indoor_Smoke_Exposure', 'Tobacco_Marketing_Exposure'
]

NUMERICAL_COLUMNS = [
    'Age', 'Mortality_Risk', '5_Year_Survival_Probability'
]

ORDINAL_COLUMNS = [
    'Air_Pollution_Exposure'  # Low, Medium, High
]

TARGET_COLUMN = 'Final_Prediction'