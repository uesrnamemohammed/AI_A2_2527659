## regression
For test row 4 the model predicted 11.65 t, but the real yield was 8.91 t, so the prediction was too high by 2.74 t. This is the largest error in the test set. The data contains random noise that no model can predict, so some farms will always be far from the prediction.

## classification
The model missed 3 high-risk farms (rows 99, 4 and 119). Their predicted probabilities (0.30, 0.12 and 0.35) were below 0.5, so they were labelled as not high risk. These farms are probably close to the median yield-per-hectare limit, where small random changes can flip the label. There were no false alarms.

## cluster
Cluster 3 is the largest group (41 farms): cool temperature, small fields and low rainfall. Officials can use these groups to see quickly which farms have similar problems and plan solutions for them together. The groups are not sharply separated (silhouette about 0.15), so they are only a rough guide.