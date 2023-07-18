import numpy as np
import util

from linear_model import LinearModel

def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)
    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path,add_intercept=False)
    # *** START CODE HERE ***
    gda = GDA()
    gda.fit(x_train,y_train)
    #Plotting training dataset
    util.plot(x_train,y_train,gda.theta,'output/p01e_train{}.png'.format(pred_path[-5]))
    #Predicting and plotting evaluating dataset
    eval_predict = gda.predict(x_eval)
    util.plot(x_eval,y_eval,gda.theta,'output/p01e_eval{}.png'.format(pred_path[-5]))
    #Save predictions
    np.savetxt(pred_path, eval_predict > 0.5, fmt='%d')
    # *** END CODE HERE ***

class GDA(LinearModel):

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.
        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m,n = x.shape
        self.theta = np.zeros(n+1)
        #Calculating all the parameters.
        phi = np.sum(y==1) / m #Scalar
        sum_of_1 = (y==1).reshape(m,1) * x #Shape --> (m,n)
        # (y==1) creates a 1d vector matrix of m elements with 1 if the corresponding y entry is 1 and 0 if corresponding y entry is 0
        sum_of_0 = (y==0).reshape(m,1) * x
        mu_0 =  np.sum(sum_of_0, axis=0) / np.sum(y==0) #Shape --> (m,)
        mu_1 = np.sum(sum_of_1, axis=0) / np.sum(y==1) #Shape --> (m,)
        sigma = ((sum_of_0 - mu_0).T.dot(sum_of_0 - mu_0) + (sum_of_1 - mu_1).T.dot(sum_of_1 - mu_1))/m #Shape (m,m)
        #Calculating theta. 
        sigma_inv = np.linalg.inv(sigma)
        self.theta[0] = (mu_0).T.dot(sigma_inv).dot(mu_0) - (mu_1).T.dot(sigma_inv).dot(mu_1) - np.log((1-phi)/phi)
        self.theta[1:] = sigma_inv.dot((mu_1 - mu_0))
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.
        Args:
            x: Inputs of shape (m, n).
        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return (1/(1+np.exp(-(self.theta[0]+x.dot(self.theta[1:])))))        
        # *** END CODE HERE ***
