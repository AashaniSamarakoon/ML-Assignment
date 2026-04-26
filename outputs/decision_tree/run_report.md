# Decision Tree Run Report

- Dataset file: hotel_bookings.csv
- Raw dataset shape: (119390, 32)
- Modeling dataset shape: (87230, 37)
- Best CV score: 0.817422
- Best parameters: {'classifier__min_samples_split': 40, 'classifier__min_samples_leaf': 2, 'classifier__max_depth': 24, 'classifier__criterion': 'gini', 'classifier__class_weight': None, 'classifier__ccp_alpha': 0.0001}
- Test accuracy: 0.822481
- Test precision: 0.699136
- Test recall: 0.623282
- Test F1 score: 0.659033
- Test ROC-AUC: 0.879062
- Test MCC: 0.541196

## Critical analysis prompts
- Review whether the model overfits by comparing CV performance with test performance.
- Explain how the preprocessing steps influenced model behavior.
- Discuss what future feature engineering or threshold tuning could improve recall or MCC.