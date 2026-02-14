# Modern ES Starter (Hopper-v4)

This is a modernized fork of the OpenAI Evolution Strategies starter code. It has been updated to run on **macOS (Apple Silicon)** using **Python 3.10+** and **TensorFlow 2.15+**.

## 🚀 Key Modernizations
- **Environment:** Migrated from `gym` to `gymnasium` (Hopper-v4).
- **Architecture:** Compatibility fixes for Apple Silicon M1/M2/M3 chips.
- **Visualization:** Included `record_hopper.py` for headless MP4 generation.

## 📦 Setup
### 1. Prerequisites
You need the Redis server installed to handle communication between the master and workers.
```bash
# Install using Homebrew
brew install redis
brew services start redis
```
### 2. Installation
You need the Redis server installed to handle communication between the master and workers.
```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME

# Install dependencies
pip install -r requirements.txt

# Critical fix for macOS multiprocessing
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
```

## 🏃 Running the Experiment
Open two separate terminal tabs to run the distributed training:
### **Step 1: Start the Master**
```bash
python -m es_distributed.main master --exp_str configurations/hopper.json
```
### **Step 2: Start the Workers**
```bash
python -m es_distributed.main workers --num_workers 4
```
### **Step 3: Record Progress**
Run this script to generate an MP4 video of the current training snapshot in the ./videos folder:
```bash
python record_hopper.py
```

## 📊 Results
The Hopper begins learning stable hopping patterns within ~40 iterations. As training progresses, the EpRewMean will increase, resulting in faster and more stable movement.

##
*Original research by OpenAI (2017). Modernized by Rhoad Ghunaim.*
