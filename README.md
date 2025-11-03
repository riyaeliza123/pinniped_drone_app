# Pinniped census app
Computer vision project to count number of pinnipeds in each drone image.
294 images used for training from Campbell, Nanimo and Cowichan (Vancouver Island, Canada). Total 618 images after pre-proessing for training. 

Model metrics:
- Precision: 88.6%
- Recall: 80.1%
- Accuracy: 82.9%
- False positive rate: ~11 %
- Area under ROC curve = 0.83,  indicates very good model performance (0.8 – 0.9 = strong discrimination ability)

Confusion matrix:
| | **Predicted Positive** | **Predicted Negative** | **Total Actual** |
|-------------------------|--------------|--------------|------------------|
| **Actual Positive**     | TP = 284     | FN = 71      | 355              |
| **Actual Negative**     | FP = 37      | TN = 227     | 264              |
| **Total Predicted**     | 321          | 298          | 618              |


