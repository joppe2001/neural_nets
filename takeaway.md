🔄 Basic Machine Learning Model Structure 🔄

[Training Phase]
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│    1. Input Data          2. Model                3. Output      │
│    ┌─────────┐           ┌─────────┐            ┌─────────┐      │
│    │ X₁      │           │         │            │ Y_hat   │      │
│    │ X₂      │─────────▶ │  f(x)   │──────────▶ │(predict)│      │
│    │ X₃      │           │         │            │         │      │
│    └─────────┘           └─────────┘            └────┬────┘      │
│         ▲                     ▲                      │           │
│         │                     │                      │           │
│    ┌─────────┐           ┌──────────┐           ┌────▼────┐      │
│    │Features │           │Weights & │           │ Actual  │      │
│    │         │           │Parameters│      ┌────│   Y     │      │
│    └─────────┘           └──────────┘      │    └─────────┘      │
│                               ▲            │                     │
│                               │            │                     │
│                         ┌─────────────┐    │                     │
│                         │    Loss     │    │                     │
│                         │  Function   │◀───┘                     │
│                         └─────────────┘                          │
│                               │                                  │
│                               ▼                                  │
│                         ┌─────────────┐                          │
│                         │Optimization │                          │
│                         │(Backprop)   │                          │
│                         └─────────────┘                          │
└──────────────────────────────────────────────────────────────────┘

[Inference Phase]
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│    1. New Input         2. Trained Model        3. Prediction    │
│    ┌─────────┐           ┌─────────┐            ┌─────────┐      │
│    │ X₁      │           │         │            │         │      │
│    │ X₂      │─────────▶ │  f(x)   │──────────▶ │ Y_hat   │      │
│    │ X₃      │           │         │            │         │      │
│    └─────────┘           └─────────┘            └─────────┘      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

Key Components:
1. Input Layer (Features)
   - Raw data input
   - Feature engineering/preprocessing
   
2. Model Architecture
   - Weights and parameters
   - Activation functions
   - Layer structure

3. Output Layer
   - Predictions (Y_hat)
   - Classification/Regression results

4. Training Components
   - Loss function
   - Optimization algorithm
   - Backpropagation
   - Parameter updates

5. Evaluation Metrics
   - Accuracy
   - Precision/Recall
   - MSE/MAE
   
Flow:
1. Data ➔ 2. Model ➔ 3. Prediction ➔ 4. Compare with Truth
➔ 5. Calculate Loss ➔ 6. Update Parameters ➔ Repeat