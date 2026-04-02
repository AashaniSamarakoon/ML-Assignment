# Random Forest Run Report

- Dataset file: hotel_bookings.csv
- Raw dataset shape: (119390, 32)
- Modeling dataset shape: (87230, 37)
- Best CV score: 0.897396
- Best parameters: {'classifier__n_estimators': 500, 'classifier__min_samples_split': 5, 'classifier__min_samples_leaf': 2, 'classifier__max_features': 0.5, 'classifier__max_depth': None, 'classifier__class_weight': 'balanced_subsample'}
- Test accuracy: 0.842715
- Test precision: 0.714197
- Test recall: 0.714494
- Test F1 score: 0.714345
- Test ROC-AUC: 0.903950
- Test MCC: 0.605827

## Critical analysis prompts
- Review whether the model overfits by comparing CV performance with test performance.
- Explain how the preprocessing steps influenced model behavior.
- Discuss what future feature engineering or threshold tuning could improve recall or MCC.