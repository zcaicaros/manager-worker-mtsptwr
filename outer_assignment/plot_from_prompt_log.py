import numpy as np
import matplotlib.pyplot as plt
import re

path = './'

f = open(path+'mtsptwr_log.txt', 'r')
logs = f.readlines()

training_log = []
validation_log = []
for log in logs:
    # find decimals each log line
    digits = re.findall(r'\d*\.?\d+', log)
    if len(digits) > 2:  # if is a training log
        training_log.append(np.array(digits[1:-1]).astype(float))
    elif len(digits) == 2:  # else validation log
        validation_log.append(np.array(digits).astype(float))
    else:
        continue  # useless log line

# arrange result
validation_log = np.stack(validation_log)
training_log = np.stack(training_log)


# plot validation log...
validation_rej = validation_log[:, 0]
validation_len = validation_log[:, 1]
x_validation = np.array([i for i in range(validation_log.shape[0])])
plt.xlabel('Validation')
plt.ylabel('Avg.Rej_Rate')
plt.plot(x_validation, validation_rej, 'k', color='tab:blue')
plt.grid()
plt.show()
plt.xlabel('Validation')
plt.ylabel('Avg.Max_subtour_length')
plt.plot(x_validation, validation_len, 'k', color='tab:blue')
plt.grid()
plt.show()


# plot validation log...
training_rej = training_log[:, 1]
training_len = training_log[:, 2]
x_validation = np.array([i for i in range(training_log.shape[0])])
plt.xlabel('Training Itr.')
plt.ylabel('Avg.Rej_Rate')
plt.plot(x_validation, training_rej, 'k', color='tab:blue')
plt.grid()
plt.show()
plt.xlabel('Training Itr.')
plt.ylabel('Avg.Max_subtour_length')
plt.plot(x_validation, training_len, 'k', color='tab:blue')
plt.grid()
plt.show()
