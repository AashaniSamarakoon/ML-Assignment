# Model Comparison Report

- Dataset file: hotel_bookings.csv
- Test set size: 17446
- Best overall model: Random Forest

## Ranking table
| model_name          |   accuracy |   balanced_accuracy |   precision |   recall |   specificity |       f1 |       f2 |   roc_auc |   average_precision_pr_auc |      mcc |   cohen_kappa |   brier_score |   log_loss |
|:--------------------|-----------:|--------------------:|------------:|---------:|--------------:|---------:|---------:|----------:|---------------------------:|---------:|--------------:|--------------:|-----------:|
| Random Forest       |   0.842715 |            0.802952 |    0.714197 | 0.714494 |      0.891411 | 0.714345 | 0.714434 |  0.90395  |                   0.787467 | 0.605827 |      0.605827 |      0.1106   |   0.35401  |
| Gradient Boosting   |   0.83876  |            0.782981 |    0.7292   | 0.658892 |      0.907071 | 0.692266 | 0.671848 |  0.903149 |                   0.77973  | 0.584783 |      0.583407 |      0.110495 |   0.344502 |
| Decision Tree       |   0.780007 |            0.799278 |    0.567659 | 0.842149 |      0.756406 | 0.678182 | 0.767887 |  0.880451 |                   0.715514 | 0.543893 |      0.520505 |      0.145832 |   0.444065 |
| Logistic Regression |   0.765906 |            0.776763 |    0.551477 | 0.800916 |      0.75261  | 0.653193 | 0.734474 |  0.847889 |                   0.664884 | 0.504703 |      0.485436 |      0.162052 |   0.490945 |

## Discussion prompts
- Compare performance trade-offs between interpretability and accuracy.
- Discuss whether the best ROC-AUC model is also the best MCC or recall model.
- Use the saved plots in outputs/comparison to support the report discussion.