# Logistic Regression Run Report

- Dataset file: hotel_bookings.csv
- Raw dataset shape: (119390, 32)
- Modeling dataset shape: (87230, 37)
- Best CV score: 0.839518
- Best parameters: {'classifier__C': 0.5, 'classifier__class_weight': 'balanced', 'classifier__solver': 'liblinear'}
- Test accuracy: 0.765964
- Test precision: 0.551556
- Test recall: 0.800916
- Test F1 score: 0.653248
- Test ROC-AUC: 0.847889
- Test MCC: 0.504787

## Critical analysis prompts
- Review whether the model overfits by comparing CV performance with test performance.
- Explain how the preprocessing steps influenced model behavior.
- Discuss what future feature engineering or threshold tuning could improve recall or MCC.