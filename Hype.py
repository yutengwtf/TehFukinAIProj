MINIBATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 180000
TARGET_NETWORK_UPDATE_FREQUENCY = 8000
DISCOUNT_FACTOR = 0.99
UPDATE_FREQUENCY = 4
LEARNING_RATE = 0.00025
GRADIENT_MOMENTUM = 0.95
SQUARED_GRADIENT_MOMENTUM = 0.95
MIN_SQUARED_GRADIENT = 0.01
INITIAL_EXPLORATION = 1
FINAL_EXPLORATION = 0.1
FINAL_EXPLORATION_FRAME = 1000000
REPLAY_START_SIZE = 50000

### 自己加的
TOTAL_TRAINING_FRAME = 7000000
GAMEPLAY_BEFORE_START = 45

### 第一練
START_FRAME1 = 1
END_FRAME1 = 3240000
START_EPS1 = 1
END_EPS1 = 0.618
TOTAL1 = 4320000

### 第二練
START_FRAME2 = 1
MID_FRAME = 540000
END_FRAME = 3240000
START_EPS2 = 1
MID_EPS2 = 0.618
END_EPS2 = 0.382
TOTAL2 = 4320000

### 最後一練
START_FRAME3 = 1 
MID1_FRAME3 = 1080000
MID2_FRAME3 = 3240000 
END_FRAME3 = 6480000 
EPS_START3 = 1 
MID1_EPS3 = 0.618 
MID2_EPS3 = 0.382 
END_EPS3 = 0.05
TOTAL3 = 7020000

SAVE_FRAME = 90000
