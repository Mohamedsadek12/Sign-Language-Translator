# First attempt
## First stage: data uploading and augmentation
- ### params used:
    - size = 64 pixels
    - batch = 32
    - 0-1 scale (instead of 0-255)
    - validation data is 20%
    - rotation range = 10
    - width shift range = 0.1
    - height shift range = 0.1
    - zoom range = 0.1
    - brightness range = 0.8-1.2
    - no horizontal flip 
- ### generating the dataset:
  - one hot encoding
  - shuffle used
## Second stage: attempt a custom CNN first
  - optimizer = adam
  - loss = categorical cross entropy
  - metrics = accuracy
  - used dropout layers between hidden layers
  - used batch normalization between layers
  - train first model with 10 epochs