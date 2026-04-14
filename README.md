# Self-Driving Car Simulation Using CNN

## Project Overview
This project implements an end-to-end convolutional neural network (CNN) to predict the steering angle of a self-driving car using front camera images from a simulation environment.

The system follows a hybrid approach:
- A CNN performs perception by predicting steering angles from images
- A rule-based controller manages throttle based on speed and steering

The trained model is deployed in real time using the Udacity self-driving car simulator.

---

## Dataset
The dataset was collected using the simulator in training mode by manually driving the car.

Each sample includes:
- Center, left, and right camera image paths
- Steering angle
- Throttle, brake, and speed

For this project:
- Input: Center camera images
- Output: Steering angle

---

## Data Preprocessing

### Dataset Balancing
The dataset initially contained a large number of near-zero steering values (straight driving), which introduced bias.

To address this:
- A histogram of steering angles was created
- Samples per bin were limited using `max_samples`
- Excess samples were removed

Final setting:
- `max_samples = 250`

This provided a balanced distribution while maintaining realistic driving behavior.

---

### Image Preprocessing
Each image undergoes the following steps:
- Crop to remove sky and car hood
- Convert color space (BGR to YUV)
- Apply Gaussian blur
- Resize to (200 × 66)
- Normalize pixel values

These steps ensure the model focuses on relevant road features and improves training stability.

---

### Data Augmentation
To improve generalization:
- Horizontal flipping (with inverted steering)
- Random brightness adjustment
- Random zoom

Augmentation is applied only to training data.

---

## Model Architecture
A CNN inspired by the Nvidia end-to-end driving model was implemented:
- 5 convolutional layers for feature extraction
- Flatten layer
- Fully connected layers for regression
- Output layer predicting a single steering value

Key configurations:
- Activation: ELU
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam

---

## Training Process
- Data shuffled to reduce bias
- Split into training and validation sets (80/20)
- Batch generator used for memory efficiency and on-the-fly augmentation

Training configuration:
- Batch size: 32
- Epochs: up to 50
- Early stopping (patience = 10)

---

## Training Experiments

### Summary of Runs
Multiple experiments were conducted to analyze performance:

- Incorrect preprocessing significantly reduced performance
- Undertraining resulted in poor feature learning
- Overtraining caused overfitting and unstable driving
- Excessive augmentation introduced noise
- Dataset balance strongly influenced turning behavior

### Best Model (Run 13)
Settings:
- `max_samples = 250`
- `epoch = 50`
- `steps_per_epoch = 150`
- `validation_steps = 100`
- `patience = 10`
- `bins = 25`
- `steps_per_epoch=len(X_train) // 32`
- `validation_steps=len(X_test) // 32`

- BGR to YUV conversion
- Removed noisy augmentations

Results:
- Stable lane following
- Good handling of curves
- Best balance of generalization and stability

---

## Simulation and Deployment

The trained model is deployed using a Socket.IO server.

The simulator provides:
- Camera images (base64 encoded)
- Speed data

The system returns:
- Steering angle (CNN prediction)
- Throttle (rule-based control)

---

## Steering Control
The raw steering output is adjusted before sending to the simulator:
- The steering value is slightly increased to reduce understeering  
- The steering is smoothed using the previous steering value to avoid sudden changes  
- The steering angle is limited to prevent sharp turns  

These adjustments help the car drive more smoothly and improve overall stability.

---

## Throttle Control
Throttle is controlled using simple rules:
- During sharp turns, the car slows down  
- On straight paths, the car speeds up  
- If the car goes faster than the target speed, it slows down 

This helps maintain stable and controlled driving.
---

## Results
The final model successfully:
- Follows the road
- Navigates turns
- Maintains stable driving behavior

Performance improvements were achieved through:
- Dataset balancing
- Data augmentation
- Steering smoothing
- Dynamic speed control

---

## Limitations
- Inconsistent normalization between training and inference
- Validation set used during training (no separate test set)
- Throttle not learned by the model

---

## Future Improvements
- Keep preprocessing consistent between training and testing  
- Allow the model to predict both steering and speed  
- Improve how speed is controlled for smoother driving  
- Use side camera images to improve recovery when the car drifts off track

---

## How to Run

### Train the Model
python model.py

### Run Simulation
python TestSimulation.py

### Start Simulator
- Open Udacity simulator
- Select leftmost track
- Select autonomous mode

---

## Conclusion
This project shows how a CNN can be used for end-to-end steering prediction in a self-driving car simulation. By combining deep learning for perception with simple rule-based control, the system is able to achieve stable and effective driving behavior.

## Video demonstration
https://youtu.be/SeGHe6kXgaE
