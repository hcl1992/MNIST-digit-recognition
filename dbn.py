from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from nolearn.dbn import DBN
from sklearn.metrics import classification_report, accuracy_score

def train_test_prep():
    '''
    This function will load the MNIST data, scale it to a 0 to 1 range, and split it into test/train sets. 
    '''

    image_data = fetch_mldata('MNIST Original') # Get the MNIST dataset.

    basic_x = image_data.data
    basic_y = image_data.target # Separate images from their final classification. 

    min_max_scaler = MinMaxScaler() # Create the MinMax object.
    basic_x = min_max_scaler.fit_transform(basic_x.astype(float)) # Scale pixel intensities only.

    x_train, x_test, y_train, y_test = train_test_split(basic_x, basic_y, 
                            test_size = 0.2, random_state = 0) # Split training/test.
    return x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test = train_test_prep()

dbn_model = DBN([x_train.shape[1], 500, 10], 
                learn_rates = 0.2, 
                learn_rate_decays = 0.9,
                dropouts = 0.25, # Express the percentage of nodes that will be randomly dropped as a decimal.
                epochs = 50) 

dbn_model.fit(x_train, y_train)
y_true, y_pred = y_test, dbn_model.predict(x_test) # Get our predictions
print(classification_report(y_true, y_pred)) # Classification on each digit
print 'The accuracy is:', accuracy_score(y_true, y_pred)
