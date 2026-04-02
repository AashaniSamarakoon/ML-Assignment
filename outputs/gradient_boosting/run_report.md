# Gradient Boosting Run Report

- Dataset file: hotel_bookings.csv
- Raw dataset shape: (119390, 32)
- Modeling dataset shape: (87230, 37)
- Best CV score: 0.896374
- Best parameters: {'classifier__min_samples_leaf': 30, 'classifier__max_iter': 400, 'classifier__max_depth': None, 'classifier__max_bins': 255, 'classifier__learning_rate': 0.1, 'classifier__l2_regularization': 0.01}
- Test accuracy: 0.838760
- Test precision: 0.729200
- Test recall: 0.658892
- Test F1 score: 0.692266
- Test ROC-AUC: 0.903149
- Test MCC: 0.584783

## Critical analysis prompts
- Review whether the model overfits by comparing CV performance with test performance.
- Explain how the preprocessing steps influenced model behavior.
- Discuss what future feature engineering or threshold tuning could improve recall or MCC.