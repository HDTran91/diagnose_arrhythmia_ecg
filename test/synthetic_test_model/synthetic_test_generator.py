
import numpy as np
import pandas as pd

def generate_heart_rate(rhythm):
    if "Normal" in rhythm:
        return np.random.uniform(60, 100)   # Example: Random heart rate for normal rhythm
    elif "Atrial Flutter" in rhythm:
        return np.random.uniform(100, 175)  # Example: Random heart rate for tachycardia
    else:
        return np.random.uniform(40, 60) 
    

def generate_synthetic_data(unique_patterns, num_samples=1000):
   
    synthetic_data = []
    for _, pattern in unique_patterns.iterrows():
        rhythm = str(pattern['rhythm_classification'])
        synthetic_samples = {
            'rhythm': np.repeat(rhythm, num_samples),
            'age': np.random.randint(20, 80, num_samples),  # Random age between 20 and 80
            'sex': np.random.choice([0, 1], num_samples),  # 0 for Male, 1 for Female
            'heart_rate': [generate_heart_rate(rhythm) for _ in range(num_samples)],  # Generate heart rates based on rhythm
            'weight': np.random.randint(40, 150, num_samples),  # Random weight
            'standard_deviation': np.random.uniform(0.01, 0.2, num_samples)  # Random standard deviation
        }
        synthetic_data.append(pd.DataFrame(synthetic_samples))

    # Combine all synthetic data into a single DataFrame
    synthetic_test_set = pd.concat(synthetic_data, ignore_index=True)

    # Shuffle the synthetic test set
    synthetic_test_set = synthetic_test_set.sample(frac=1).reset_index(drop=True)

    return synthetic_test_set