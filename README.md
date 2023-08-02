# Prediciting High-Risk Pregnancy in Mississippi Delta
This repo contains code analyzing home visit data for Delta Health Alliance (ETO) and [national birth data from CDC](https://www.cdc.gov/nchs/data_access/vitalstatsonline.htm#Births). Two algorithm were chosen: 
1. combination of logistic regression and decision (80.52% accuracy, 63.51% recall)
2. Gradient Boosted Machine (96.62% accuracy, 96% recall)
## Groups of ETO Data 
- Background + prenatal form (pre_preg only)
- Background + prenatal form + parent/child (with birth data)
- Background + prenatal form + parent/child + home visit data
 
## Algorithms 
- Logistic regression – with and without SMOTE
- Decision tree – with and without SMOTE
- Random forest
- GBM