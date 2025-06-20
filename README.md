#  Spacecraft Collision Avoidance System

This project predicts the risk of collision between spacecraft and space debris using CDM (Conjunction Data Message) data. I aim to implement a neural-network-based solution for proactive risk prediction and avoidance maneuvers.

---

## Dataset Used

The dataset is provided by the **European Space Agency (ESA)** as part of the  
[**Spacecraft Collision Avoidance Challenge**](https://kelvins.esa.int/collision-avoidance-challenge/data/).  
It contains time series of CDMs (Conjunction Data Messages) representing potential collision events between spacecraft and objects in orbit, collected between **2015–2019**.

The dataset includes:
- Over **15000 conjunction events**
- Around **200,000 CDM entries**
- 103 attributes per CDM (e.g., time to TCA, miss distance, position uncertainty, risk estimate, etc.)

### Feature Selection

Picking the features that have strong predictive power is important to avoid learning noise in the data and overfitting on the training set.  
Therefore, **7 out of 103 features** are selected based on domain relevance and prior research:

- `time_to_tca`
- `max_risk_estimate`
- `max_risk_scaling`
- `mahalanobis_distance`
- `miss_distance`
- `c_position_covariance_det`
- `c_obs_used`

These features serve as the input vector to the initial model and help reduce dimensionality while preserving predictive performance.
For these 7 feature, the attributes come from the latest available CDM. 
