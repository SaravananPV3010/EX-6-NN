<H3>Saravanan PV</H3>
<H3>212223230195</H3>
<H3>EX. NO.6</H3>
<H3>27.04.2025</H3>
<H1 ALIGN =CENTER>Heart attack prediction using MLP</H1>
<H3>Aim:</H3>  To construct a  Multi-Layer Perceptron to predict heart attack using Python
<H3>Algorithm:</H3>
Step 1:Import the required libraries: numpy, pandas, MLPClassifier, train_test_split, StandardScaler, accuracy_score, and matplotlib.pyplot.<BR>
Step 2:Load the heart disease dataset from a file using pd.read_csv().<BR>
Step 3:Separate the features and labels from the dataset using data.iloc values for features (X) and data.iloc[:, -1].values for labels (y).<BR>
Step 4:Split the dataset into training and testing sets using train_test_split().<BR>
Step 5:Normalize the feature data using StandardScaler() to scale the features to have zero mean and unit variance.<BR>
Step 6:Create an MLPClassifier model with desired architecture and hyperparameters, such as hidden_layer_sizes, max_iter, and random_state.<BR>
Step 7:Train the MLP model on the training data using mlp.fit(X_train, y_train). The model adjusts its weights and biases iteratively to minimize the training loss.<BR>
Step 8:Make predictions on the testing set using mlp.predict(X_test).<BR>
Step 9:Evaluate the model's accuracy by comparing the predicted labels (y_pred) with the actual labels (y_test) using accuracy_score().<BR>
Step 10:Print the accuracy of the model.<BR>
Step 11:Plot the error convergence during training using plt.plot() and plt.show().<BR>
<H3>Program: </H3>

```python
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
data = pd.read_csv('heart.csv')

x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

x_train , x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

mlp = MLPClassifier(hidden_layer_sizes=(100,100),max_iter=1000,random_state=42)
training_loss = mlp.fit(x_train,y_train).loss_curve_

y_pred = mlp.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

plt.plot(training_loss)
plt.title("MLP Training Loss Convergence")
plt.xlabel("Iteration")
plt.ylabel("Training loss")
plt.show()

```
```python
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import pandas as pd

heart_data = pd.read_csv('heart.csv')
x = heart_data.drop('target',axis=1)
y = heart_data['target']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

mlp_classifier = MLPClassifier(hidden_layer_sizes=(64,),max_iter=1000,random_state=42)
x_train_scaled = scaler.fit_transform(x_train)
mlp_classifier.fit(x_train_scaled,y_train)

y_pred = mlp_classifier.predict(x_test_scaled)
accuracy = accuracy_score(y_test,y_pred)
conf_matrix = confusion_matrix(y_test,y_pred)
classification_rep = classification_report(y_test,y_pred)
print(f"Accuracy:{accuracy}")
print("\n Confusion Matrix:")
print(conf_matrix)
print("Classification Report")
print(classification_rep)
```
## OUTPUT:

![Screenshot 2025-04-27 022327](https://github.com/user-attachments/assets/d953bd37-57b3-4b97-9400-d0798dcc75a3)
![Screenshot 2025-04-27 022340](https://github.com/user-attachments/assets/16094a85-2772-4bb4-a1d6-ad353232a0ec)





<H3>Results:</H3>
Thus, an ANN with MLP is constructed and trained to predict the heart attack using python.
