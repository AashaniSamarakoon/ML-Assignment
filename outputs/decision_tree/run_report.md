# Decision Tree Run Report

- Dataset file: hotel_bookings.csv
- Raw dataset shape: (119390, 32)
- Modeling dataset shape: (87230, 37)
- Best CV score: 0.873321
- Best parameters: {'classifier__min_samples_split': 2, 'classifier__min_samples_leaf': 5, 'classifier__max_depth': 16, 'classifier__criterion': 'gini', 'classifier__class_weight': 'balanced', 'classifier__ccp_alpha': 0.0001}
- Test accuracy: 0.780007
- Test precision: 0.567659
- Test recall: 0.842149
- Test F1 score: 0.678182
- Test ROC-AUC: 0.880451
- Test MCC: 0.543893

## Critical analysis prompts
- Review whether the model overfits by comparing CV performance with test performance.
- Explain how the preprocessing steps influenced model behavior.
- Discuss what future feature engineering or threshold tuning could improve recall or MCC.